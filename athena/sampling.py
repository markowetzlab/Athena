import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from multiprocessing import Pool, RLock

class Sampling:
    
    def sample(self, ncells=10000, pop_fp=None, sim_fp=None, cache=True, return_data=False):
        print (f"Simulation: {self.network_name} Sampling Cells...", flush=True)
        cells_meta, gene_expr = self.sampling_cells(ncells, sim_fp)
        
        print (f"Simulation: {self.network_name} Sampling Molecules...", flush=True)
        lib_sizes = self.sampling_molecules(gene_expr, pop_fp)
        cells_meta = self.clean_cells_metadata(cells_meta, lib_sizes)
        cells_meta = cells_meta.reset_index(drop=True)
        
        if cache:
            print (f"Simulation: {self.network_name} Caching....", flush=True)
            cells_meta.to_csv(os.path.join(self.metadata_dir,
                                           'cells_metadata.csv.gz'),
                              compression='gzip', index=False)
        
        if return_data:
            fp = os.path.join(self.metadata_dir, 'gene_expression.csv.gz')
            gene_expr = pd.read_csv(fp, dtype=np.int16)
            return cells_meta, gene_expr
        else:
            return None, None
        
    def sampling_cells(self, ncells, sim_fp):
        
        if sim_fp is None:
            sim_fp = os.path.join(self.results_dir, 'simulated_counts.csv.gz')
        
        self.cell_sim_meta = pd.read_csv(f'{self.results_dir}/cell_metadata.csv.gz')
        self.cell_sim_meta = self.cell_sim_meta.reset_index().rename(columns={'index': 'cell_i'})
        
        if ncells > self.cell_sim_meta.shape[0]:
            raise Exception(f"Simulation: {self.network_name} Number of cells requested is greater than the number of cells simulated. Sample fewer cells...")
            
        cells_meta = []
        cells = np.array([i for i in range(self.cell_sim_meta.shape[0])])
        cells_meta = self.get_cells_meta()
        cells = self.sample_cells_per_grna(cells_meta, ncells)
        
        cells_meta = cells_meta.iloc[cells]
        gene_expr = self.load_cells(cells, sim_fp)
        
        return cells_meta, gene_expr
    
    def sampling_molecules(self, gene_expr, pop_fp=None):
        
        if pop_fp is None:
            pop = sc.read_loom(self.pop_fp)
        else:
            pop = sc.read_loom(pop_fp)
        
        realcounts = pop.X.toarray()
        cell_umi = pop.obs.total_counts.values
        
        lib_size = self.calc_library_size(cell_umi, gene_expr)
        self.downsampling(realcounts, gene_expr, lib_size)
        
        return lib_size
    
    def clean_cells_metadata(self, meta, lib_sizes):
        meta['lib_size'] = lib_sizes
        meta['grna'] = meta['sim_label'].apply(lambda x: "_".join(x.split('_')[0:2]))
        meta['target_gene'] = meta['sim_label'].apply(lambda x: x.split('-grna')[0])
        
        if self.crispr_type == 'knockout':
            meta['is_cell_perturbed'] = meta['sim_label'].apply(lambda x: x.split('_')[-1])
            meta.loc[meta.target_gene == self.ctrl_label, 'is_cell_perturbed'] = self.ctrl_label
        else:
            meta['is_cell_perturbed'] = 'PRT'
            meta.loc[meta.target_gene == self.ctrl_label, 'is_cell_perturbed'] = self.ctrl_label
        
        meta = meta.reset_index(drop=True)
        return meta
    
    def load_cells(self, sampled_cells, sim_fp):
        df = pd.read_csv(sim_fp, dtype=np.int16)
        df = df.iloc[sampled_cells]
        
        return df
        
    def calc_library_size(self, cell_umis, sim_counts):
        sim_counts_ls = sim_counts.sum(axis=1).values
        
        if self.map_reference_ls:
            # sampling library
            sim_probs = np.random.uniform(size=len(sim_counts_ls))
            lib_size = np.around(np.quantile(cell_umis, sim_probs))
        else:
            lib_size = sim_counts_ls
        
        return lib_size
    
    def downsampling(self, realcount, sim_counts, lib_sizes, cache_size=100000):
        gene_expr = []
        sim_cols = list(sim_counts.columns)
        real_cpm = self.get_real_cpm(realcount)
        gene_expr_fp = os.path.join(self.metadata_dir, 'gene_expression.csv.gz')
        
        if len(lib_sizes) < cache_size:
            cache_size = round(len(lib_sizes) / 2)
        
        sizes = [lib_sizes[i:i+cache_size-1] for i in range(0, len(lib_sizes), cache_size)]
        counts = [sim_counts.iloc[i:i+cache_size-1, :] for i in range(0, len(sim_counts), cache_size)]
        
        for i in tqdm(range(len(sizes))):
            df = counts[i]
            lib_size = np.array(sizes[i])
            cpm = df / lib_size.reshape(-1, 1)
            cpm = self.calc_cpm(cpm, real_cpm)
            
            # sample molecules
            for index, size in enumerate(lib_size):
                gene_val = cpm[index, ]
                gene_expr = np.random.multinomial(size, gene_val)
                cpm[index, ] = gene_expr
            
            
            gene_expr = pd.DataFrame(cpm, columns=sim_cols, dtype=np.int16)
            self.cache_dataframe(gene_expr, gene_expr_fp)
    
    def get_real_cpm(self, realcount):
        # calculating realcount datasets cpm
        real_ls = np.sum(realcount, axis=1).reshape(-1, 1)
        real_cpm = realcount / real_ls
        real_cpm = real_cpm.flatten()
        real_cpm = real_cpm[real_cpm != 0]
    
        return real_cpm
    
    def calc_cpm(self, scpm, rcpm):
        
        if self.map_reference_cpm:
             # sort sim counts data via least to greatest
            rcpm = rcpm.flatten()
            sim_shape = scpm.shape
            scpm_size = sim_shape[0] * sim_shape[1]
            
            probs = np.random.uniform(size=scpm_size)
            scpm = np.quantile(rcpm, probs).reshape(sim_shape)
            scpm = scpm / np.sum(scpm, axis=1).reshape(-1, 1)
        
        return scpm
    
    def sample_cells_per_grna(self, cells_meta, ncells):
        sampled_cells = []
        ngrnas = len(self.sim_meta.grna.unique())
        self.ncells_per_grna = round(ncells / ngrnas)
        
        for row_i in range(len(self.sim_meta)):
            row = self.sim_meta.iloc[row_i]
            sim_cells = cells_meta.loc[cells_meta.sim_label == row.sim_name, 'cell_i'].values
            sim_cells = list(sim_cells)
            
            if len(sim_cells) < self.ncells_per_grna:
                print ("changing ncells_per_grna...")
                self.ncells_per_grna = len(sim_cells)            

            if row.sample_percent != 0:
                n = int(self.ncells_per_grna * row.sample_percent)
                sampled = random.sample(sim_cells, k=n)
                sampled_cells = sampled_cells + sampled
        
        return sampled_cells
    
    def get_cells_meta(self):
        cells_meta = []
        ncells_per_sim = int(self.perturb_time / self.update_interval)

        for row_i, row in self.sim_meta.iterrows():
            nsims_adjust = 1 + self.sim_meta.nsims.iloc[:row_i].sum()

            for row_sim_i in range(row.nsims):
                row_sim_i = row_sim_i + nsims_adjust
                cell_sim_meta = self.cell_sim_meta.loc[self.cell_sim_meta.sim_i == row_sim_i]

                for cell_i in range(cell_sim_meta.shape[0]):
                    cells_meta.append({"cell_i": cell_sim_meta.iloc[cell_i].cell_i,
                                       "sim_i": cell_sim_meta.iloc[cell_i].sim_i,
                                       "sim_label": row.sim_name, 
                                       "grna_label": row.grna})
                    
        return pd.DataFrame(cells_meta)

    def cache_dataframe(self, df, fp):
        
        if os.path.exists(fp):
            df.to_csv(fp, mode='a', index=False, header=False, compression='gzip')
        else:
            df.to_csv(fp, index=False, compression='gzip')
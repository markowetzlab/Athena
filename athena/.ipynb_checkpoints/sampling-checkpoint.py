import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from multiprocessing import Pool, RLock

class Sampling:
    
    def sample(self, ncells=10000, pop_fp=None, cache=True, sim_fp=None):
        print (f"Simulation: {self.network_name} Sampling Cells...", flush=True)
        cells_meta, gene_expr = self.sampling_cells(ncells, sim_fp)
        print (f"Simulation: {self.network_name} Sampling Molecules...", flush=True)
        gene_expr, lib_sizes = self.sampling_molecules(gene_expr, pop_fp)
        
        cells_meta = self.clean_cells_metadata(cells_meta, lib_sizes)
        cells_meta = cells_meta.reset_index(drop=True)
        
        if cache:
            print (f"Simulation: {self.network_name} Caching....", flush=True)
            cells_meta.to_csv(os.path.join(self.metadata_dir, 'cells_metadata.csv.bz2'), compression='bz2', index=False)
            gene_expr.to_csv(os.path.join(self.metadata_dir, 'gene_expression.csv.bz2'), compression='bz2', index=False)
        
        return cells_meta, gene_expr
        
    def sampling_cells(self, ncells, sim_fp):
        
        if sim_fp is None:
            sim_fp = os.path.join(self.results_dir, 'simulated_counts.csv.gz')
        
        ncells_from_sim = (self.perturb_time / self.update_interval)
        max_cells = int(self.sim_meta.nsims.sum() * ncells_from_sim)
        
        if ncells > max_cells:
            raise Exception(f"Simulation: {self.network_name} Number of cells requested is greater than the number of cells simulated. Sample fewer cells...")
            
        cells_meta = []
        cells = np.array([i for i in range(max_cells)])
        sims, sim_labels = self.get_cells_sims()
        
        for sim, sim_label, cell in tqdm(zip(sims, sim_labels, cells)):
            cells_meta.append({"cell_label": f"cell_{cell}", "sim_label": sim_label, "cell_i": cell,
                               "sim_i": sim, "fp": os.path.join(self.results_dir, f'simulation_{sim}.csv')})
        
        cells_meta = pd.DataFrame(cells_meta)
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
        
        lib_size = self.calc_library_size(cell_umi, gene_expr.to_numpy(dtype=np.int16))
        simcounts_cpm = self.calc_cpm(realcounts, gene_expr.to_numpy(dtype=np.int16))
        downsampled_counts = self.downsampling(simcounts_cpm, lib_size)
        gene_expr = pd.DataFrame(downsampled_counts, columns=gene_expr.columns)
        
        return gene_expr, lib_size
    
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
        sim_counts_ls = np.sum(sim_counts, axis=1)
        
        if self.map_reference_ls:
            # sampling library
            sim_probs = np.random.uniform(size=len(sim_counts_ls))
            lib_size = np.around(np.quantile(cell_umis, sim_probs))
        else:
            lib_size = sim_counts_ls
        
        return lib_size
    
    def calc_cpm(self, realcount, sim_counts):
        realcount_ls = np.sum(realcount, axis=1).reshape(-1, 1)
        sim_counts_ls = np.sum(sim_counts, axis=1).reshape(-1, 1)
        
        realcount_cpm = realcount / realcount_ls
        sim_counts_cpm = sim_counts / sim_counts_ls
        
        if self.map_reference_cpm:
            # sort sim counts data via least to greatest
            sim_shape = sim_counts.shape
            realcount_cpm = realcount_cpm.flatten()
            sim_counts_cpm = sim_counts_cpm.flatten()
            
            realcount_cpm = realcount_cpm[realcount_cpm != 0]
            sim_probs = np.random.uniform(size=len(sim_counts_cpm))
            sim_counts_cpm = np.quantile(realcount_cpm, sim_probs).reshape(sim_shape)
            sim_counts_cpm = sim_counts_cpm / np.sum(sim_counts_cpm, axis=1).reshape(-1, 1)
        
        return sim_counts_cpm
    
    def downsampling(self, sim_counts_cpm, lib_sizes):
        
        for index, lib_size in tqdm(enumerate(lib_sizes)):
            gene_val = sim_counts_cpm[index, ]
            gene_expr = np.random.multinomial(lib_size, gene_val)
            sim_counts_cpm[index, ] = gene_expr
        
        return sim_counts_cpm
    
    def sample_cells_per_grna(self, cells_meta, ncells):
        sampled_cells = []
        ngrnas = len(self.sim_meta.grna.unique())
        self.ncells_per_grna = round(ncells / ngrnas)
        
        for row_i in range(len(self.sim_meta)):
            row = self.sim_meta.iloc[row_i]
            sim_cells = cells_meta.loc[cells_meta.sim_label == row.sim_name, 'cell_i'].values
            
            if row.sample_percent != 0:
                n = int(self.ncells_per_grna * row.sample_percent)
                sampled = random.choices(sim_cells, k=n)
                sampled_cells = sampled_cells + sampled
        
        return sampled_cells
    
    def get_cells_sims(self):
        cell_sim, sim_labels = [], []
        ncells_per_sim = int(self.perturb_time / self.update_interval)
        max_cells = int(self.sim_meta.nsims.sum() * ncells_per_sim)

        for row_i, row in self.sim_meta.iterrows():
            nsims_adjust = 1 + self.sim_meta.nsims.iloc[:row_i].sum()
            
            for row_sim_i in range(row.nsims):
                row_sim_i = row_sim_i + nsims_adjust
                sims = [row_sim_i] * ncells_per_sim
                
                for sim_i in sims:
                    sim_labels.append(row.sim_name)
                    cell_sim.append(sim_i)
                    
        return cell_sim, sim_labels
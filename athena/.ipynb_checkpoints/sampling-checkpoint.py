import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from multiprocessing import Pool, RLock

class Sampling:
    
    def sample(self, ncells=10000, pop_fp=None, cache=True):
        print ("Sampling Cells...")
        cells_meta, gene_expr, cells = self.sampling_cells(ncells)
        print ("Sampling Molecules...")
        gene_expr = self.sampling_molecules(gene_expr, pop_fp)
        gene_expr['cells'] = cells
        
        if cache:
            print ("Caching....")
            cells_meta.to_csv(os.path.join(self.metadata_dir, 'cells_metadata.csv'), index=False)
            gene_expr.to_csv(os.path.join(self.metadata_dir, 'gene_expression.csv'), index=False)
        
        return cells_meta, gene_expr
        
    def sampling_cells(self, ncells=10000):
        ncells_from_sim = (self.perturb_time / self.update_interval)
        max_cells = int((self.sim_meta.shape[0] * self.nsims_per_condition) * ncells_from_sim)
        
        if ncells > max_cells:
            raise Exception("Number of cells requested is greater than the number of cells simulated. Sample fewer cells...")
        

        cells_meta = []
        cells = np.array([i for i in range(max_cells)])
        sims = (cells // ncells_from_sim).astype(int)

        for sim, cell in tqdm(zip(sims, cells)):
            grna_label = self.multiplier_df.index[sim]
            cells_meta.append({"cell_label": f"cell_{cell}", "sim_label": grna_label, "cell_i": cell,
                               "sim_i": sim, "fp": os.path.join(self.results_dir, f'simulation_{sim}.csv')})
        
        cells_meta = pd.DataFrame(cells_meta)
        
        if self.crispr_type == 'knockout':
            cells = self.ko_sample_cells(cells_meta, ncells)
        else:
            cells = list(random.choices(cells, k=ncells))
        
        cells_meta = cells_meta.iloc[cells]
        gene_expr = self.collapse_molecules(cells)
        
        return cells_meta, gene_expr, cells
        
    def sampling_molecules(self, gene_expr, pop_fp=None):
        
        if pop_fp is None:
            pop = sc.read_loom(self.pop_fp)
        else:
            pop = sc.read_loom(pop_fp)
        
        realcounts = pop.X.toarray()
        cell_umi = pop.obs.total_counts.values
        
        lib_size = self.calc_library_size(cell_umi, gene_expr.values)
        simcounts_cpm = self.calc_cpm(realcounts, gene_expr.values)
        downsampled_counts = self.downsampling(simcounts_cpm, lib_size)
        gene_expr = pd.DataFrame(downsampled_counts, columns=gene_expr.columns)
        
        return gene_expr
        
    def collapse_molecules(self, sampled_cells):
        df = pd.read_csv(os.path.join(self.results_dir, 'simulated_counts.csv'))
        df = df.sort_values(by=['sim_i'])
        df = df.reset_index(drop=True)
        df = df.iloc[sampled_cells]
        df = df.drop(columns=['sim_i'])
        
        if self.collapse_mrna:
            spec = pd.read_csv(os.path.join(self.species_vec_dir, 'batch_0.csv'))
            spec = spec.drop(columns=['species', 'state', 'sim_i'])
            spec = spec.loc[spec.molecule_type != 'protein', ]
            spec = spec.drop_duplicates(subset=['spec_name'])
            spec = spec.sort_values(by=['gene'])

            mrna = spec.loc[spec.molecule_type == 'mrna', 'spec_name'].values
            premrna = spec.loc[spec.molecule_type == 'premrna', 'spec_name'].values
            df = pd.DataFrame(df[mrna].values + df[premrna].values, columns=mrna)
        
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
    
    def ko_sample_cells(self, cells_meta, ncells):
        sampled_cells = []
            
        for row_i in range(len(self.sim_meta)):
            row = self.sim_meta.iloc[row_i]
            sim_cells = cells_meta.loc[cells_meta.sim_label == row.sim_name, 'cell_i'].values
            
            n = int(ncells * row.sample_percent)
            sampled = random.choices(sim_cells, k=n)
            sampled_cells = sampled_cells + sampled
        
        return sampled_cells
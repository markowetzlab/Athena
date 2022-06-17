import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from multiprocessing import Pool, RLock
from numpy.random import uniform, multinomial

class Sampling:
    
    def sample(self, ncells=10000, pop_fp=None, cache=True, sim_fp=None):
        print (f"Simulation: {self.network_name} Sampling Cells...", flush=True)
        cells_meta, gene_expr = self.sampling_cells(ncells, sim_fp)
        
        print (f"Simulation: {self.network_name} Sampling Molecules...", flush=True)
        gene_expr, lib_sizes = self.sampling_molecules(gene_expr, pop_fp)
        cells_meta = self.clean_cells_metadata(cells_meta, lib_sizes)
        
        if cache:
            print (f"Simulation: {self.network_name} Caching....", flush=True)
            cells_meta.to_csv(os.path.join(self.metadata_dir, 'cells_metadata.csv.bz2'), compression='bz2', index=False)
            gene_expr.to_csv(os.path.join(self.metadata_dir, 'gene_expression.csv.bz2'), compression='bz2', index=False)
        
        return cells_meta, gene_expr
        
    def sampling_cells(self, ncells, sim_fp):
        meta_fp = os.path.join(self.results_dir, 'cell_metadata.csv.gz')
        
        if sim_fp is None:
            sim_fp = os.path.join(self.results_dir, 'simulated_counts.csv.gz')
        
        cells_meta = pd.read_csv(meta_fp)\
                       .reset_index()\
                       .rename(columns={"index": "cell_i"})

        if ncells > cells_meta.shape[0]:
            raise Exception(f"Simulation: {self.network_name} Number of cells requested is greater than the number of cells simulated. Sample fewer cells...")
        
        cells_meta = self.sample_cells_per_grna(cells_meta, ncells)
        gene_expr = self.load_cells(cells_meta, sim_fp)
        
        return cells_meta, gene_expr
    
    def sampling_molecules(self, gene_expr, pop_fp=None):
        
        if pop_fp is None:
            pop = sc.read_loom(self.pop_fp)
        else:
            pop = sc.read_loom(pop_fp)
        
        realcounts = pop.X.toarray()
        cell_umi = pop.obs.total_counts.values
        
        lib_size = self.calc_library_size(cell_umi, gene_expr)
        simcounts_cpm = self.calc_cpm(realcounts, gene_expr)
        downsampled_counts = self.downsampling(simcounts_cpm, lib_size)
        gene_expr = pd.DataFrame(downsampled_counts, columns=gene_expr.columns, dtype=np.int16)
        
        return gene_expr, lib_size
    
    def clean_cells_metadata(self, meta, lib_sizes):
        meta['lib_size'] = lib_sizes
        meta['target_gene'] = meta['sim_name'].apply(lambda x: x.split('-grna')[0])
        
        if self.crispr_type == 'knockout':
            meta['is_cell_perturbed'] = meta['sim_name'].apply(lambda x: x.split('_')[-1])
            meta.loc[meta.target_gene == self.ctrl_label, 'is_cell_perturbed'] = self.ctrl_label
        else:
            meta['is_cell_perturbed'] = 'PRT'
            meta.loc[meta.target_gene == self.ctrl_label, 'is_cell_perturbed'] = self.ctrl_label
        
        meta = meta.reset_index(drop=True)
        return meta
    
    def load_cells(self, cells_meta, sim_fp):
        df = pd.read_csv(sim_fp, dtype=np.int16)
        df = df.iloc[cells_meta.cell_i.values]
        
        return df
        
    def calc_library_size(self, cell_umis, sim_counts):
        
        if self.map_reference_ls:
            # sampling library
            nlibs = sim_counts.shape[0]
            sim_probs = uniform(size=nlibs).astype(np.float16)
            lib_size = np.around(np.quantile(cell_umis, sim_probs))
        else:
            lib_size = sim_counts.sum(axis=1).values
        
        return lib_size
    
    def calc_cpm(self, realcount, sim_counts):
        realcount_ls = np.sum(realcount, axis=1).reshape(-1, 1)
        sim_counts_ls = sim_counts.sum(axis=1).values.reshape(-1, 1)
        
        realcount_cpm = realcount / realcount_ls
        sim_counts_cpm = sim_counts / sim_counts_ls
        
        if self.map_reference_cpm:
            # sort sim counts data via least to greatest
            sim_shape = sim_counts.shape
            realcount_cpm = realcount_cpm.flatten()
            sim_cpm_size = sim_shape[0] * sim_shape[1]
            
            realcount_cpm = realcount_cpm[realcount_cpm != 0]
            
            print ('Sampling Probabilities...', flush=True)
            sim_probs = uniform(size=sim_cpm_size).astype(np.float16)
            
            print ('Fetching Quantiles...', flush=True)
            sim_counts_cpm = np.quantile(realcount_cpm, sim_probs).astype(np.float16).reshape(sim_shape)
            sim_counts_cpm = sim_counts_cpm / np.sum(sim_counts_cpm, axis=1).astype(np.float16).reshape(-1, 1)
        
        return sim_counts_cpm
    
    def downsampling(self, sim_counts_cpm, lib_sizes):
        print ('Downsampling Counts...', flush=True)
        
        for index, lib_size in tqdm(enumerate(lib_sizes)):
            gene_val = sim_counts_cpm[index, ]
            gene_val = gene_val.astype(np.float64)
            gene_val = gene_val / sum(gene_val)
            
            gene_expr = multinomial(lib_size, gene_val).astype(np.int16)
            sim_counts_cpm[index, ] = gene_expr        
        
        return sim_counts_cpm
    
    def sample_cells_per_grna(self, cells_meta, ncells):
        sampled_cells = []
        grnas = cells_meta.grna.unique()
        ncells_per_grna = round(ncells / len(grnas))
        
        for row_i in range(len(self.sim_meta)):
            row = self.sim_meta.iloc[row_i]
            sim_cells = cells_meta.loc[cells_meta.sim_name == row.sim_name, 'cell_i'].values
            sim_cells = list(sim_cells)
            
            if len(sim_cells) < ncells_per_grna:
                print ("changing ncells_per_grna...")
                ncells_per_grna = len(sim_cells)            

            if row.sample_percent != 0:
                n = int(ncells_per_grna * row.sample_percent)
                sampled = random.sample(sim_cells, k=n)
                sampled_cells = sampled_cells + sampled
        
        cells_meta = cells_meta.iloc[sampled_cells]
        return cells_meta
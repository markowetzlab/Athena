import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools as it

class GuideRNA:
    
    """
        Issues with specificity score is not an one to one mapping in simulations as compared to real world data.
    """
    
    def generate_grnas(self, grna_meta=None, crispr_type=None, on_target=None, off_target=None):
        
        self.check_crispr_type(crispr_type)
        
        if grna_meta is None:
            self.get_target_genes()
            self.check_target_scores(on_target, off_target)
            ngrnas_seq = range(self.ngrnas_per_target)
            ctrl_grnas = [f'{self.ctrl_label}-grna.{i + 1}' for i in ngrnas_seq]
            gene_grnas = [f'{gene}-grna.{i + 1}' for gene in self.target_genes for i in ngrnas_seq]

            grna_names = ctrl_grnas + gene_grnas
            ngrnas = len(grna_names)
            
            grna_meta = self.get_perturbed_genes(grna_names)
            self.grna_meta = self.sample_on_target(grna_meta)
        else:
            meta = []
            ngrnas = grna_meta.shape[0]
            grna_names = list(grna_meta.grna.values)
            on_target = grna_meta.on_target.unique()
            off_target = grna_meta.off_target.unique()
            
            for index, row in grna_meta.iterrows():
                grna_names = [row['grna']]
                
                if index == 0:
                    grna_names = grna_names + [f'{self.ctrl_label}-grna.1']
                
                df = self.get_perturbed_genes(grna_names, row['on_target'], row['off_target'])
                meta.append(pd.DataFrame(df))
            
            self.grna_meta = pd.concat(meta)
                
        self.ngrnas = ngrnas
        
        if self.crispr_type == 'activation':
            self.grna_meta['on_target'] = (1 - self.grna_meta['on_target']) + 1
        
        self.create_multiplier_matrix()
        self.grna_meta.to_csv(os.path.join(self.metadata_dir, 'grna_meta.csv'), index=False)
        self.sim_meta.to_csv(os.path.join(self.metadata_dir, 'simulation_meta.csv'), index=False)
        
    def get_target_genes(self):
        
        if len(self.target_genes) == 0:
            
            if self.perturb_tfs:
                tfs = list(self.feature_info.feature_id.loc[self.feature_info.is_tf])
                self.target_genes = self.target_genes + tfs
            
            if self.perturb_kinases:
                kinases = list(self.feature_info.feature_id.loc[self.feature_info.is_kinase])
                self.target_genes = self.target_genes + kinases
    
    def create_multiplier_matrix(self):
        
        if self.crispr_type == 'knockout':
            multiplier = self.create_knockout_meta()
            self.sim_meta = self.calc_sample_perc(multiplier, self.grna_meta, self.ctrl_label)
        else:
            multiplier = self.create_nonko_meta()
            self.sim_meta = pd.DataFrame({'sim_name': multiplier.index, 'grna': multiplier.index, 'sample_percent': 1})
        
        multi = []
        prev_batch = 0
        nrows = len(multiplier)
        self.sim_meta['nsims'] = np.ceil(self.nsims_per_condition * self.sim_meta['sample_percent'])
        self.sim_meta['nsims'] = self.sim_meta['nsims'].astype(int)
        self.nsims_per_condition = self.sim_meta.loc[self.sim_meta.grna == 'CTRL-grna.1', 'nsims'].values[0]
        self.calc_sims_per_device(nrows)
        
        for i in tqdm(range(self.sim_meta.shape[0])):
            grna = multiplier.index.values[i]
            row = self.sim_meta.iloc[i].copy()
            perturb_vec = list(multiplier.iloc[i].T.values)
            
            if i == 0:
                adjust_interval = 0
            else:
                adjust_interval = self.sim_meta.nsims.iloc[:i].sum()
            
            for sim_i in range(row.nsims):
                sim_i += 1 + adjust_interval
                current_batch = sim_i // self.nsims_per_device
                
                sim_meta = {"feature_id": [], "perturbation": [],
                            "sim_i": [], "sim_name": [], "grna": []}
                feature = list(multiplier.columns.values + f"_{sim_i}") 
                
                sim_meta["feature_id"] = feature
                sim_meta["perturbation"] = perturb_vec
                sim_meta["grna"] = [grna] * len(feature)
                sim_meta["sim_i"] = [sim_i] * len(feature)
                sim_meta["sim_name"] = [row.sim_name] * len(feature)
                
                multi.append(pd.DataFrame(sim_meta))
                
                if prev_batch != current_batch:
                    self.cache_multiplier(multi, prev_batch)
                    multi, prev_batch = [], current_batch
        
    def get_perturbed_genes(self, grna_names, on_target=None, off_target=None):
        perturbed_genes = []
        
        if on_target is None:
            on_target = self.on_target
        
        for grna in grna_names:
            sampled_genes = []
            
            if self.ctrl_label in grna:
                sampled_genes.append({'grna': grna, 'perturbed_gene': self.ctrl_label, 'on_target': 1, 'target': False})
            else:
                target_gene = grna.split('-')[0]
                sampled_genes = self.sample_off_target(target_gene, grna, off_target)
                sampled_genes.append({'grna': grna, 'perturbed_gene': target_gene, 'on_target': on_target, 'target': True})
            
            perturbed_genes = perturbed_genes + sampled_genes
        
        return perturbed_genes
    
    def sample_off_target(self, target_gene, grna, off_target=None):
        genes_to_add = []
        perturb_activity = 0
        info = self.feature_info
        
        if off_target is None:
            off_target = self.off_target
        
        if off_target != 1:
            ngenes = random.sample([1,2,3,4,5], k=1)[0]
            genes = random.sample(list(info.loc[info.feature_id != target_gene,'feature_id'].values), k=ngenes)

            for index, gene in enumerate(genes):

                if (index + 1) == ngenes:
                    perturb_activity = 1 - off_target
                else:
                    perturb_activity = np.random.uniform(high=off_target, size=1)[0]
                    perturb_activity = round(perturb_activity, 2)
                    
                    off_target = round(off_target - perturb_activity, 2)
                    perturb_activity = round(1 - perturb_activity, 2)

                if perturb_activity != 1:
                    genes_to_add.append({'grna': grna, 
                                         'perturbed_gene': gene,
                                         'on_target': perturb_activity,
                                         'target': False})
        
        return genes_to_add
    
    def sample_on_target(self, grna_meta):
        grna_meta = pd.DataFrame(grna_meta)
        probs = self.grna_library.probability
        on_target_scores = self.grna_library.on_target / 100
        
        nsamples = grna_meta.loc[grna_meta.on_target.isna()].shape[0]
        grna_meta.loc[grna_meta.on_target.isna(), 'on_target'] = random.choices(on_target_scores,
                                                                                weights=probs,
                                                                                k=nsamples)
        
        return grna_meta
    
    def create_nonko_meta(self):
        grnas = self.grna_meta.grna.unique()
        genes = self.feature_info['feature_id'].values
        multiplier = pd.DataFrame(data=1, index=grnas, columns=genes)
        
        for index, grna in enumerate(grnas):

            if not 'CTRL' in grna:
                grna_df = self.grna_meta.loc[self.grna_meta.grna == grna, ]
                
                for _, row in grna_df.iterrows():
                    col_index = None
                    
                    for i, col in enumerate(multiplier.columns):
                        if row.perturbed_gene == col:
                            col_index = i
                        
                    if col_index is None:
                        print (row.perturbed_gene)
                        
                    multiplier.iloc[index, col_index] = row.on_target
            
        return multiplier
    
    def create_knockout_meta(self):
        multiplier = []
        genes = self.feature_info['feature_id'].values
        
        for grna in self.grna_meta.grna.unique():
            if self.ctrl_label in grna:
                multi = pd.DataFrame(data=1, index=[grna], columns=genes)
                multiplier.append(multi)
                
            else:
                grna_df = self.grna_meta.loc[self.grna_meta.grna == grna, ]
                keys, values = zip(*{gene: [0, 1] for gene in grna_df.perturbed_gene.values}.items())
                ko_combo = pd.DataFrame([dict(zip(keys, v)) for v in  it.product(*values)])

                sims_label = self.create_sim_labels(grna, ko_combo)
                multi = pd.DataFrame(data=1, index=sims_label, columns=genes)
                multiplier.append(self.update_multiplier(multi, ko_combo))

        return pd.concat(multiplier)
    
    def create_sim_labels(self, grna, ko_combo):
        sims_label = []

        for row in range(len(ko_combo)):
            label = grna

            for col in range(len(ko_combo.columns)):
                gene = ko_combo.columns[col]
                prtb = ko_combo.iloc[row, col]

                if prtb == 0:
                    label = f'{label}_{gene}_PRT'
                else:
                    label = f'{label}_{gene}_NT'

            sims_label.append(label)

        return sims_label

    def update_multiplier(self, grna_multi, ko_combo):
        
        for col in ko_combo.columns:
            grna_multi[col] = ko_combo[col].values

        return grna_multi
    
    def calc_sample_perc(self, multiplier, grna_meta, ctrl_label):
        grnas = []
        sample_perc = []

        for row_i, rowname in enumerate(multiplier.index):
            grna_label = "_".join(rowname.split("_")[:2])
            grna = grna_meta.loc[grna_meta.grna == grna_label].copy()
            ngenes = grna.shape[0]
            grnas.append(grna_label)

            if ctrl_label in rowname:
                sample_perc.append(1)
                
            elif ngenes == 1:
                if 'PRT' in rowname:
                    if grna.on_target.values[0] == 0:
                        sample_perc.append(1)
                    else:
                        sample_perc.append(grna.on_target.values[0])
                else:
                    if grna.on_target.values[0] == 0:
                        sample_perc.append(0)
                    else:
                        sample_perc.append(1 - grna.on_target.values[0])
                
            else:
                perc = self.multiple_genes_percentage(grna, multiplier, row_i)
                sample_perc.append(perc)

        sim_meta = pd.DataFrame({'sim_name': multiplier.index, 'grna': grnas, 'sample_percent': sample_perc})
        sim_meta = sim_meta.loc[sim_meta.sample_percent != 0]
        return sim_meta

    def multiple_genes_percentage(self, grna_meta, multiplier, sim_name):
        grna_meta['perturbed_gene'] = grna_meta['perturbed_gene'].values
        ko_combos = multiplier[list(grna_meta['perturbed_gene'].values)].iloc[sim_name]
        
        for col in ko_combos.index:
            value = ko_combos[col]
            gene = grna_meta.loc[grna_meta.perturbed_gene == col, 'perturbed_gene'].values[0]
            gene_on_target = grna_meta.loc[grna_meta.perturbed_gene == col, 'on_target'].values[0]

            if value == 1:
                # probability of no edit
                ko_combos[col] = gene_on_target
            else:
                # probability of edit
                ko_combos[col] = 1 - gene_on_target
                
        return round(np.prod(ko_combos), 2)
    
    def calc_sims_per_device(self, nrows):
        nsims = 0
        total_sims = self.sim_meta.nsims.sum()
        ngrnas = len(self.sim_meta.grna.unique())
        
        while nsims != total_sims:
            self.nsims_per_device = total_sims // self.nbatches
            nsims = self.nsims_per_device * self.nbatches
            
            if nsims != total_sims:
                self.nbatches += 1
            
    def check_crispr_type(self, crispr_type):
        
        try:
            crispr_type = crispr_type.lower()
            if crispr_type in ['interference', 'activation', 'knockout']:
                self.crispr_type = crispr_type
            else:
                raise Exception("crispr_type parameter must be either: activation, interference, or knockout...")
        except:
            if self.crispr_type is None:
                raise Exception("crispr_type parameter must be either: activation, interference, or knockout...")
                
    def check_target_scores(self, on_target, off_target):
        probs = self.grna_library.probability
        
        if (not on_target is None) and (on_target <= 1) and (on_target >= 0):
            self.on_target = float(on_target)
        
        if (not off_target is None) and (off_target <= 1) and (off_target >= 0):
            self.off_target = float(off_target)
                
        if self.off_target is None:
            self.off_target = random.choices(self.grna_library.off_target, weights=probs, k=1)[0]
        
        if self.on_target is None:
            self.on_target = random.choices(self.grna_library.on_target, weights=probs, k=1)[0]
            
    def cache_multiplier(self, multi, prev_batch):
        fp = os.path.join(self.multiplier_dir, f'batch_{prev_batch}.parquet')
        multi = pd.concat(multi, ignore_index=True)
        multi.to_parquet(fp, compression='brotli')
        
        
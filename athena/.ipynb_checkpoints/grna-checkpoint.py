import os
import random
import pandas as pd
from tqdm import tqdm

class GuideRNA:
    
    def generate_grnas(self):
        self.get_target_genes()
        ngrnas_seq = range(self.ngrnas_per_target)
        ctrl_grnas = [f'{self.ctrl_label}-grna.{i + 1}' for i in ngrnas_seq]
        gene_grnas = [f'{gene}-grna.{i + 1}' for gene in self.target_genes for i in ngrnas_seq]
        
        grna_names = ctrl_grnas + gene_grnas
        self.ngrnas = len(grna_names)
        grna_meta = self.get_perturbed_genes(grna_names)
        grna_meta = self.sample_on_target(grna_meta)
        
        self.grna_meta = pd.DataFrame(grna_meta)
        
        if self.crispr_type == 'activation':
            self.grna_meta['on_target'] = self.grna_meta['on_target'] + 1
        
        self.create_multiplier_matrix()
    
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
        
        nrows = len(multiplier)
        multiplier.columns = "transcription_" + multiplier.columns.values
        self.nsims_per_device = (self.nsims_per_condition * nrows) // self.nbatches
        multi = [multiplier.iloc[i,] for i in range(nrows) for sim_i in range(self.nsims_per_condition)]
        
        self.multiplier = pd.concat(multi)
        self.multiplier_df = pd.DataFrame(multi)
        self.multiplier_df.index = self.multiplier_df.index.set_names('grna_label')
        self.multiplier_df.to_csv(os.path.join(self.metadata_dir, 'multiplier.csv'))
    
    def get_perturbed_genes(self, grna_names):
        perturbed_genes = []
        
        for grna in tqdm(grna_names):
            sampled_genes = []
            
            if self.ctrl_label in grna:
                sampled_genes.append({'grna': grna, 'perturbed_gene': self.ctrl_label, 'on_target': 1, 'target': False})
            else:
                target_gene = grna.split('-grna')[0]
                sampled_genes = self.sample_off_target(target_gene, grna)
                sampled_genes.append({'grna': grna, 'perturbed_gene': target_gene, 'on_target': self.on_target / 100, 'target': True})
            
            perturbed_genes = perturbed_genes + sampled_genes
        
        return perturbed_genes
    
    def sample_off_target(self, target_gene, grna):
        genes_to_add = []
        genes = list(self.feature_info.feature_id.values)
        off_target_genes = random.sample(genes, k=self.off_target)
        
        for index, gene in enumerate(off_target_genes):
            if gene == target_gene:
                continue
           
            genes_to_add.append({'grna': grna, 'perturbed_gene': gene, 'on_target': 0, 'target': False})
            
        if len(genes_to_add) != self.off_target:
            genes_to_add = self.sample_off_target(target_gene, grna)
        
        return genes_to_add
    
    def sample_on_target(self, grna_meta):
        probs = self.grna_library.probability
        on_target_scores = self.grna_library.on_target / 100
        
        for index, meta in enumerate(grna_meta):
            
            if not meta['target'] and meta['perturbed_gene'] != self.ctrl_label:
                meta['on_target'] = random.choices(on_target_scores, weights=probs, k=1)[0]
            
            grna_meta[index] = meta
        
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
                    grna.on_target.values[0]
                    sample_perc.append(grna.on_target.values[0])
                else:
                    sample_perc.append(1 - grna.on_target.values[0])
            else:
                perc = self.multiple_genes_percentage(grna, multiplier, row_i)
                sample_perc.append(perc)

        sim_meta = pd.DataFrame({'sim_name': multiplier.index, 'grna': grnas, 'sample_percent': sample_perc})
        return sim_meta

    def multiple_genes_percentage(self, grna_meta, multiplier, sim_name):
        grna_meta['perturbed_gene'] = grna_meta['perturbed_gene'].values
        ko_combos = multiplier[list(grna_meta['perturbed_gene'].values)].iloc[sim_name]

        for col in ko_combos.index:
            value = ko_combos[col]
            gene_on_target = grna_meta.loc[grna_meta.perturbed_gene == col, 'on_target'].values[0]

            if value == 1:
                # probability of grna edit
                ko_combos[col] = 1 - gene_on_target
            else:
                # probability of no edit
                ko_combos[col] = gene_on_target

        return round(np.prod(ko_combos), 2)
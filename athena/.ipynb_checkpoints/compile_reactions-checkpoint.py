import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, RLock

class CompileReactions:
    
    reactions = {"transcription_": "None",
                 "splicing_": "mol_premrna_",
                 "translation_": "mol_mrna_",
                 "premrna_decay_": "mol_premrna_",
                 "mrna_decay_": "mol_mrna_",
                 "protein_decay_": "mol_protein_",
                 "phosphorylation_": "mol_protein_",
                 "dephosphorylation_": "mol_phospho_protein_",
                 "phospho_protein_decay_": "mol_phospho_protein_"}
    
    def compiling(self):
        
        with Pool(processes=self.ncpus, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
            jobs = [pool.apply_async(self.compile_reactions, args=(batch_i,)) for batch_i in range(self.nbatches)]
            pool.close()
            
            results = [job.get() for job in jobs]      
    
    def compile_reactions(self, device_i):
        file = f'batch_{device_i}.csv'
        
        print (f"Starting to Processing Batch {device_i}...")
        noise_info, noise_network, regulators = self.create_duplicates(device_i)

        start = device_i * len(noise_info)
        end = start + len(noise_info)
        
        species_vec = self.create_species_vector(noise_info, device_i)
        propensity = self.create_propensity_matrix(noise_info, self.multiplier.iloc[start:end], species_vec, file)
        affinity = self.create_affinity_matrix(noise_network, propensity, species_vec)
        change_vec = self.create_change_vector(noise_info, propensity, species_vec)
        
        affinity.to_csv(os.path.join(self.affinity_dir, file), index=False)
        regulators.to_csv(os.path.join(self.regulators_dir, file), index=False)
        propensity.to_csv(os.path.join(self.propensity_dir, file), index=False)
        change_vec.to_csv(os.path.join(self.change_vec_dir, file), index=False)
        species_vec.to_csv(os.path.join(self.species_vec_dir, file), index=False)
        print (f"Finished Processing Batch {device_i}...")
    
    def create_duplicates(self, device_i):
        noise_info = pd.concat([self.feature_info.copy()] * self.nsims_per_device).reset_index()
        noise_network = pd.concat([self.feature_network.copy()] * self.nsims_per_device).reset_index()
        
        # setting up species names for downstream processes and calculating simuluation number
        noise_info['sim_i'] = (noise_info.index.values // self.feature_info.shape[0] + 1) + (device_i * self.nsims_per_device)
        noise_network['sim_i'] = (noise_network.index.values // self.feature_network.shape[0] + 1) + (device_i * self.nsims_per_device)
        
        noise_network['to'] = noise_network['to'] + '_' + noise_network['sim_i'].astype(str)
        noise_network['from'] = noise_network['from'] + '_' + noise_network['sim_i'].astype(str)
        noise_info['feature_id'] = noise_info['feature_id'] + '_' + noise_info['sim_i'].astype(str)
        
        noise_info = self.inject_rate_noise(noise_info)
        noise_info, regulators = self.get_reactions_regulators(noise_network, noise_info)
        noise_network = self.inject_interaction_noise(noise_network, noise_info)
        
        kin_bool = noise_network.kinase_edge
        eff_bool = noise_network.effect == 1
        
        grn = noise_network.loc[~kin_bool, ].copy()
        phospho = noise_network.loc[(kin_bool) & (eff_bool), ].copy()
        dephospho = noise_network.loc[(kin_bool) & (~eff_bool), ].copy()
        phospho_tfs = noise_info.loc[(noise_info.is_tf) & (noise_info.is_phosphorylated), 'feature_id'].values
        
        grn['to'] = 'transcription_' + grn['to']
        phospho['to'] = 'phosphorylation_' + phospho['to']
        dephospho['to'] = 'dephosphorylation_' + dephospho['to']
        
        grn_bool = grn['from'].isin(phospho_tfs)
        phospho['from'] = 'mol_phospho_protein_' + phospho['from']
        dephospho['from'] = 'mol_phospho_protein_' + dephospho['from']
        grn.loc[~grn_bool, 'from'] = 'mol_protein_' + grn.loc[~grn_bool, 'from']
        grn.loc[grn_bool, 'from'] = 'mol_phospho_protein_' + grn.loc[grn_bool, 'from']
        
        noise_network = pd.concat([grn, phospho, dephospho])
        noise_info.drop(columns=['index'], inplace=True)
        noise_network.drop(columns=['index'], inplace=True)
        
        noise_info = noise_info.reset_index(drop=True)
        noise_network = noise_network.reset_index(drop=True)
        return noise_info, noise_network, regulators

    def create_species_vector(self, feature_info, device_i):
        molecules = ['premrna', 'mrna', 'protein']
        mrna = list('mol_mrna_' + feature_info.feature_id.values)
        premrna = list('mol_premrna_' + feature_info.feature_id.values) 
        protein = list('mol_protein_' + feature_info.feature_id.values)
        phospho_protein = list('mol_phospho_protein_' + feature_info.loc[feature_info.is_phosphorylated, 'feature_id'].values)
        species = premrna + mrna + protein + phospho_protein
        
        species_state = pd.DataFrame({'species': species, 'state': [0] * len(species)})
        
        species_state['gene'] = species_state['species'].apply(lambda x: '_'.join(x.split('_')[-3:-1]))
        species_state['sim_i'] = species_state['species'].apply(lambda x: [int(char) for char in x.split('_') if char.isdigit()][-1])
        species_state['molecule_type'] = species_state['species'].apply(lambda x: [char for char in x.split('_') if char in molecules][0])
        species_state['filepath'] = os.path.join(self.simulator_dir, 'species_vec', f'batch_{device_i}.csv')
        species_state['spec_name'] = species_state['species'].apply(lambda x: '_'.join(x.split("_")[:-1]))
        
        return species_state
    
    def create_propensity_matrix(self, feature_info, perturb_vec, species_vec, file):
        # need to add reversible reaction for dephosphorlyation
        propensity_dfs = []
        phosphos = feature_info.loc[feature_info.is_phosphorylated, ]
        
        for reaction in self.reactions.keys():
            col = reaction + 'rate'
            
            basal = 0
            rev_reaction = 0
            perturbation = 1
            independence = 0
            effects_sums = 0
            reaction_type = 0
            base_activity = 0

            if col == 'premrna_decay_rate':
                col = 'mrna_decay_rate'
            
            if reaction == 'transcription_':
                reaction_type = 1
                species_needed = 'None'
                perturbation = perturb_vec.values
                basal = feature_info.basal.values
                reaction_rates = feature_info[col].values
                nregulators = feature_info.nregulators.values
                effects_sums = feature_info.effects_sums.values
                independence = feature_info.independence.values
                base_activity = feature_info.base_activity.values
                reacts = reaction + feature_info.feature_id.values
            elif 'phospho' in col:
                if 'protein_decay' in col:
                    col = 'protein_decay_rate'
                    
                if 'dephospho' in col:
                    rev_reaction = 1
                
                reaction_rates = phosphos[col].values
                
                basal = phosphos.basal.values
                reaction_rates = phosphos[col].values
                independence = phosphos.independence.values
                nregulators = phosphos.nregulators.values
                effects_sums = phosphos.kinase_effects_sums.values
                base_activity = phosphos.kinase_base_activity.values
                effects_sums =  phosphos.kinase_effects_sums.values
                species_needed = self.reactions[reaction] + phosphos.feature_id.values
                reacts = reaction + phosphos.feature_id.values
            else:
                reaction_rates = feature_info[col].values
                species_needed = self.reactions[reaction] + feature_info.feature_id.values
                reacts = reaction + feature_info.feature_id.values
                
            reaction_set = pd.DataFrame({'reaction': reacts, 'reaction_rate': reaction_rates})
            
            reaction_set['basal'] = basal
            reaction_set['effects_sums'] = 0
            reaction_set['base_activity'] = 0.0
            reaction_set['species'] = species_needed
            reaction_set['nregulators'] = nregulators
            reaction_set['independence'] = independence
            reaction_set['effects_sums'] = effects_sums
            reaction_set['perturbation'] = perturbation
            reaction_set['base_activity'] = base_activity
            reaction_set['reaction_type'] = reaction_type
            reaction_set['reversible_reaction'] = rev_reaction
            
            propensity_dfs.append(reaction_set)
        
        propensity = pd.concat(propensity_dfs)
        species_index = {'index':'species_index'}
        temp = species_vec['species'].reset_index()
        propensity = propensity.merge(temp.rename(columns=species_index), on='species', how='left')
        
        propensity['species_index'] = propensity['species_index'].fillna(0)
        propensity['species_index'] = propensity['species_index'].astype(np.int32)
        
        self.num_reactions = len(propensity)
        return propensity
    
    def create_affinity_matrix(self, feature_network, propensity, species_vec): 
        temp_prop = propensity['reaction'].reset_index().rename(columns={'index': 'to_index'})
        temp_spec = species_vec['species'].reset_index().rename(columns={'index': 'from_index'})
        
        feature_network = feature_network.merge(temp_prop, left_on='to', right_on='reaction', how='left')
        feature_network = feature_network.merge(temp_spec, left_on='from', right_on='species', how='left')
        return feature_network

    def create_change_vector(self, noise_info, propensity, species_vec):
        phospho_feature = noise_info.loc[noise_info.is_phosphorylated, 'feature_id']
        change_rows = [{'species': 'mol_premrna_' + noise_info.feature_id, 
                        'reaction': "transcription_" + noise_info.feature_id,
                        'reaction_effect': [1] * len(noise_info)},
                       {'species': 'mol_mrna_' + noise_info.feature_id, 
                        'reaction': "splicing_" + noise_info.feature_id, 
                        'reaction_effect': [1] * len(noise_info)},
                       {'species': 'mol_protein_' + noise_info.feature_id,
                        'reaction': "translation_" + noise_info.feature_id, 
                        'reaction_effect': [1] * len(noise_info)},
                       {'species': 'mol_premrna_' + noise_info.feature_id,
                        'reaction': "splicing_" + noise_info.feature_id,
                        'reaction_effect': [-1] * len(noise_info)},
                       {'species': 'mol_premrna_' + noise_info.feature_id, 
                        'reaction': "premrna_decay_" + noise_info.feature_id,
                        'reaction_effect': [-1] * len(noise_info)},
                       {'species': 'mol_mrna_' + noise_info.feature_id, 
                        'reaction': "mrna_decay_" + noise_info.feature_id,
                        'reaction_effect': [-1] * len(noise_info)},
                       {'species': 'mol_protein_' + noise_info.feature_id,
                        'reaction': "protein_decay_" + noise_info.feature_id,
                        'reaction_effect': [-1] * len(noise_info)},
                       {'species': 'mol_phospho_protein_' + phospho_feature,
                        'reaction': "phosphorylation_" + phospho_feature,
                        'reaction_effect': [1] * len(phospho_feature)},
                       {'species': 'mol_protein_' + phospho_feature,
                        'reaction': "phosphorylation_" + phospho_feature,
                        'reaction_effect': [-1] * len(phospho_feature)},
                       {'species': 'mol_phospho_protein_' + phospho_feature,
                        'reaction': "dephosphorylation_" + phospho_feature,
                        'reaction_effect': [-1] * len(phospho_feature)},
                       {'species': 'mol_protein_' + phospho_feature,
                        'reaction': "dephosphorylation_" + phospho_feature,
                        'reaction_effect': [1] * len(phospho_feature)},
                       {'species': 'mol_phospho_protein_' + phospho_feature,
                        'reaction': "protein_decay_" + phospho_feature,
                        'reaction_effect': [-1] * len(phospho_feature)}]
        
        change_vec = pd.concat([pd.DataFrame(row) for row in change_rows])
        reacts = propensity[['reaction']].reset_index().rename(columns={'index':'react_index'})
        species = species_vec[['species']].reset_index().rename(columns={'index':'species_index'})

        change_vec = change_vec.merge(species, on='species', how='left')
        change_vec = change_vec.merge(reacts, on='reaction', how='left')
        return change_vec
    
    def inject_rate_noise(self, feature_info):
        nrows = feature_info.shape[0]
        feature_info['splicing_rate'] = feature_info['splicing_rate'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        feature_info['mrna_halflife'] = feature_info['mrna_halflife'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        feature_info['protein_halflife'] = feature_info['protein_halflife'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        feature_info['translation_rate'] = feature_info['translation_rate'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        feature_info['transcription_rate'] = feature_info['transcription_rate'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        feature_info['basal'] = feature_info['basal'].apply(lambda x: min(1 , x * np.random.normal(loc=self.noise_mean, scale=self.noise_std)))
        
        feature_info['mrna_decay_rate'] = math.log(2) / feature_info['mrna_halflife']
        feature_info['protein_decay_rate'] = math.log(2) / feature_info['protein_halflife']
        
        feature_info['max_premrna'] = feature_info['transcription_rate'] / (feature_info['mrna_decay_rate'] + feature_info['splicing_rate'])
        feature_info['max_mrna'] = feature_info['splicing_rate'] / feature_info['mrna_decay_rate'] * feature_info['max_premrna']
        feature_info['max_protein'] = feature_info['translation_rate'] / feature_info['protein_decay_rate'] * feature_info['max_mrna']
        return feature_info
    
    def inject_interaction_noise(self, feature_network, feature_info):
        nrows = feature_network.shape[0]
        feature_network = feature_network.drop(columns=['max_protein'])
        feature_network['hill'] = feature_network['hill'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        feature_network['strength'] = feature_network['strength'] * np.random.normal(size=nrows, loc=self.noise_mean, scale=self.noise_std)
        
        feature_network = feature_network.merge(feature_info[['feature_id', 'max_protein']].rename(columns={'feature_id': 'from'}), on='from', how='left')
        feature_network['dissociation'] = feature_network['max_protein'] / 2
        return feature_network
    
    def get_reactions_regulators(self, feature_network, feature_info):
        new_rows = []
        regulators_index = feature_network.groupby(['to', 'kinase_edge']).apply(lambda x: np.array([r for r in x.index.values]))
        number_of_regulators = feature_network.groupby(['to', 'kinase_edge']).apply(lambda x: len(x.index.values))
        
        regulators_index = regulators_index.reset_index().rename(columns={'to': 'feature_id', 0:'regulators'})
        nregulators = number_of_regulators.reset_index().rename(columns={'to': 'feature_id', 0: 'nregulators'})
        
        feature_info = feature_info.merge(regulators_index, on=['feature_id'], how='left')
        feature_info = feature_info.merge(nregulators, on=['feature_id', 'kinase_edge'], how='left')
        feature_info['nregulators'] = feature_info['nregulators'].fillna(0)
        
        for i in feature_info.index:
            if feature_info.regulators[i] is np.nan:
                feature_info.at[i, 'regulators'] = np.array([0])
        
        regulators = pd.DataFrame(np.concatenate([x.ravel() for x in feature_info.regulators.values]), columns=['regulators'])
        
        # collapse feature_info so it contains nkinase_regulators
        feature_info['nkinase_reg'] = 0
        feature_info['kinase_edge'] = feature_info['kinase_edge'].fillna(False)
        feature_info.loc[feature_info.kinase_edge, 'nkinase_reg'] = feature_info.loc[feature_info.kinase_edge, 'nregulators']
        feature_info.loc[feature_info.kinase_edge, 'nregulators'] = 0
        
        feature_info = feature_info.sort_values(['nregulators', 'nkinase_reg'], ascending=False).drop_duplicates(['feature_id'])
        
        # removing unnecessary columns
        feature_info.drop(columns=['regulators'], inplace=True)
        return feature_info, regulators
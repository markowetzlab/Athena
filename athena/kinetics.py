import os
import math
import random
import numpy as np
import pandas as pd
from numpy.random import uniform
from scipy.stats import truncnorm

class Kinetics:
    """I am concerned about the basal levels/effects. Check what previous basal levels are like from the R code."""
        
    def initialize_kinetics(self):
        self.sample_rates()
        self.sample_interactions()
        self.calc_dissociation()
        
        grn = self.feature_network.loc[~self.feature_network.kinase_edge, ]
        kinases = self.feature_network.loc[self.feature_network.kinase_edge, ]
        
        # calculate for GRN
        self.calc_basal_activity(grn)
        self.calc_effects_sums(grn)
        self.calc_base_activity()
        
        # calculate for phosphorylation
        self.calc_basal_activity(kinases, "kinase_basal")
        self.calc_effects_sums(kinases, "kinase_effects_sums")
        self.calc_base_activity("kinase_base_activity", "kinase_basal", "kinase_effects_sums")
        
        self.feature_network = self.feature_network.drop(columns=['feature_id'])
        print ("Initialized Kinetics...")
        
        if self.cache_network:
            self.feature_info.to_parquet(os.path.join(self.metadata_dir, 'feature_info.parquet'), compression='brotli')
            self.feature_network.to_parquet(os.path.join(self.metadata_dir, 'feature_network.parquet'), compression='brotli')
    
    def sample_rates(self):
        finfo = self.feature_info
        nrows = len(self.feature_info)
        nphospho = len(self.phosphorylated)
        finfo['transcription_rate'] = np.nan
        
        finfo['effects_sums'] = 0
        finfo['independence'] = 1
        finfo['base_activity'] = 0
        finfo['kinase_effects_sums'] = 0
        finfo['kinase_base_activity'] = 0
        finfo['phosphorylation_rate'] = 0
        finfo['dephosphorylation_rate'] = 0
        finfo['mrna_halflife'] = uniform(size=nrows,  high=5, low=2.5)
        finfo['protein_halflife'] = uniform(size=nrows, high=10, low=5)
        finfo['transcription_rate'] = uniform(size=nrows, high=50, low=20)
        finfo['translation_rate'] = uniform(size=nrows, high=150, low=100)
        finfo['splicing_rate'] = math.log(2) / 2
        finfo['mrna_decay_rate'] = math.log(2) / self.feature_info['mrna_halflife']
        finfo['protein_decay_rate'] = math.log(2) / self.feature_info['protein_halflife']
        finfo.loc[finfo.is_phosphorylated, 'phosphorylation_rate'] = uniform(size=nphospho, high=20, low=10)
        finfo.loc[finfo.is_phosphorylated, 'dephosphorylation_rate'] = uniform(size=nphospho, high=20, low=10)
        
        self.feature_info = finfo
        
    def sample_interactions(self):
        
        nrows = len(self.feature_network)
        fnet = self.feature_network.copy(deep=True)
        tfs = self.feature_info.loc[self.feature_info.is_tf, 'feature_id'].values
        ntf_edges = fnet.loc[fnet['from'].isin(tfs)].shape[0]
        
        fnet['basal'] = 0
        fnet['effect'] = random.choices([-1, 1], weights=[0.25, 0.75], k=nrows)
        fnet['strength'] = 10 ** uniform(size=nrows, high=2, low=1)
        fnet['hill'] = truncnorm.rvs(1, 10, loc=2, scale=2, size=nrows)
        
        self.feature_network = fnet
    
    def calc_dissociation(self):
        tfs = self.feature_info.loc[self.feature_info.is_tf, 'feature_id'].values
        remove = ["max_premrna", "max_mrna", "max_protein", "dissociation", "k", "max_protein"]
        
        finfo, fnet = self.feature_info, self.feature_network
        fnet = fnet[[col for col in fnet.columns if not col in remove]]
        finfo = finfo[[col for col in finfo.columns if not col in remove]]
        
        finfo["max_premrna"] = finfo["transcription_rate"] / (finfo["mrna_decay_rate"] + finfo["splicing_rate"])
        finfo["max_mrna"] = finfo["splicing_rate"] / finfo["mrna_decay_rate"] * finfo["max_premrna"]
        finfo["max_protein"] = finfo["translation_rate"] / finfo["protein_decay_rate"] * finfo["max_mrna"] 
        
        fnet = fnet.merge(finfo[['feature_id', 'max_protein']], left_on='from', right_on='feature_id', how='left')
        
        fnet['feature_id'] = fnet['to']
        fnet['dissociation'] = fnet['max_protein'] / 2
        self.feature_info, self.feature_network = finfo, fnet
    
    def calc_basal_activity(self, network, basal_col="basal"):
        basal_df = {"feature_id": [], basal_col: []}
        
        for index, group in network[['feature_id', 'effect']].groupby(['feature_id']):
            effects = group.effect.unique()
            basal_df["feature_id"].append(index)

            if len(effects) == 2:
                basal_df[basal_col].append(0.5)
            elif effects[0] == -1:
                basal_df[basal_col].append(1)
            elif effects[0] == 1:
                basal_df[basal_col].append(0.0001)
            else:
                basal_df[basal_col].append(np.nan)

        basal_df = pd.DataFrame(basal_df)
        self.feature_info = self.feature_info.merge(basal_df, on='feature_id', how='left')
        self.feature_info.loc[self.feature_info[basal_col].isna(), basal_col] = 1
    
    def calc_effects_sums(self, network, effects_col="effects_sums"):
        # calculating effects sums
        finfo = self.feature_info
        
        for feature, group in network.groupby(['to']):
            group = group.loc[group.effect > 0]
            
            if group.shape[0] == 0:
                pos_effects_sums = 0
            else:
                pos_effects_sums = group['effect'].sum()
            
            finfo.loc[finfo.feature_id == feature, effects_col] = pos_effects_sums
        
        self.feature_info = finfo
    
    def calc_base_activity(self, base_active_col="base_activity", basal_col="basal", effects_col="effects_sums"):
        info = self.feature_info
        above_zero = info[effects_col] > 0
        equal_or_below_zero = info[effects_col] <= 0

        info.loc[equal_or_below_zero, base_active_col] = info.loc[equal_or_below_zero, basal_col]
        info.loc[above_zero, base_active_col] = info.loc[above_zero,
                                                         basal_col] - info.loc[above_zero,
                                                                               'independence'] ** info.loc[above_zero, effects_col]
        
        info.loc[info[base_active_col].isna(), base_active_col] = 1
        self.feature_info = info
        
    def sample_interaction_effects(self, row):
        
        if 'TF' in row['from'] and 'TF' in row['to']:
            return random.choices([-1, 1], weights=[0.1, 0.9], k=1)[0]
        else:
            return random.choices([-1, 1], weights=[0.25, 0.75], k=1)[0]
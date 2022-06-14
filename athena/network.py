import os
import random
import numpy as np
import pandas as pd
import igraph as ig
from numpy.random import uniform
from .housekeeping import HouseKeeping
from .grn import GeneRegulatoryNetwork
from .modular_sampling import ModularSampling
from .signalling_cascade import SignallingCascade

    
class Network(ModularSampling, GeneRegulatoryNetwork, HouseKeeping, SignallingCascade):
    
    def create_network(self):
        
        # if precreated network is not provided create one
        if self.feature_info is None or self.feature_network is None:
            net_dfs, self.tfs, self.kinases, self.phosphorylated = [], [], [], []
            
            # load the GRN and extract the transcription factor network
            if self.verbose:
                print (f"Sampling GRN: {self.network_name}", flush=True)
            
            self.create_grn()
            
            if self.verbose:
                print (f"GRN Created: {self.network_name}", flush=True)
                
            net_dfs.append(self.get_edgelist_df(self.tf_egene_net))
            self.tfs = [f'TF_{i}' for i in range(self.num_tfs)]
            
            if self.nkinases != 0:
                if self.verbose:
                    print (f"Sampling Signalling Cascades: {self.network_name}", flush=True)
                
                self.create_signalling_cascades()
                
                if self.verbose:
                    print (f"Finished Sampling Cascades: {self.network_name}", flush=True)
                
                kinase_df = self.get_edgelist_df(self.kinases)
                self.phosphorylated = set(list(kinase_df.target.unique()) + list(kinase_df.source.unique()))
                self.kinases = [kinase for kinase in self.phosphorylated if not kinase in self.tfs]
                net_dfs.append(kinase_df)
                
            if self.num_hks != 0:
                if self.verbose:
                    print (f"Sampling HouseKeeping Network: {self.network_name}", flush=True)
                    
                self.create_hks()
                
                if self.verbose:
                    print (f"Sampled HouseKeeping Network: {self.network_name}", flush=True)
                    
                hk_df = self.get_edgelist_df(self.hks)
                self.hks = list(hk_df.source.unique()) + list(hk_df.target.unique())
                net_dfs.append(hk_df)
            
            # merged sampled GRNs
            self.merge_networks(net_dfs)
            self.create_metadata()
            print ("Created Network...", flush=True)
            
            if self.cache_network:
                self.feature_info.to_parquet(os.path.join(self.metadata_dir,
                                                          'feature_info.parquet'), 
                                             compression='brotli')
                self.feature_network.to_parquet(os.path.join(self.metadata_dir,
                                                             'feature_network.parquet'), 
                                                compression='brotli')
    
    def merge_networks(self, dfs):
        # get edgelist dataframe
        df = pd.concat(dfs)
        df = df.reset_index(drop=True)
        
        self.network = ig.Graph.DataFrame(df)
    
    def create_metadata(self):
        df_vert = self.network.get_vertex_dataframe()
        df_edge = self.network.get_edge_dataframe().reset_index(drop=True)
        
        df_vert['is_tf'] = False
        df_vert['is_hk'] = False
        df_vert['is_kinase'] = False
        df_edge['kinase_edge'] = False
        df_vert['is_phosphorylated'] = False
        
        df_edge = df_edge.drop(columns=['edge_weight'])
        df_vert = df_vert.rename(columns={'name':'feature_id'})
        df_edge = df_edge.rename(columns={'source':'from', 'target':'to'})
        df_edge['from'].replace(df_vert['feature_id'], inplace = True)
        df_edge['to'].replace(df_vert['feature_id'], inplace = True)
        
        df_vert.loc[df_vert.feature_id.isin(self.tfs), 'is_tf'] = True
        df_vert.loc[df_vert.feature_id.isin(self.hks), 'is_hk'] = True
        df_edge.loc[df_edge['from'].isin(self.kinases), 'kinase_edge'] = True
        df_vert.loc[df_vert.feature_id.isin(self.kinases), 'is_kinase'] = True
        df_vert.loc[df_vert.feature_id.isin(self.phosphorylated), 'is_phosphorylated'] = True
        
        self.feature_info = df_vert
        self.feature_network = df_edge
        
    def rename_nodes(self, network, nodes_to_rename, label):
        df_vert = network.get_vertex_dataframe()
        df_edge = network.get_edge_dataframe().reset_index(drop=True)
        nodes_to_rename = np.unique(nodes_to_rename)
        vert_loc = df_vert['name'].isin(nodes_to_rename)
                
        df_vert['name'].loc[vert_loc, ] = [f'{label}_{i}' for i in range(len(nodes_to_rename)) if nodes_to_rename[i] in df_vert['name'].values]
        df_edge['source'].replace(df_vert['name'], inplace=True)
        df_edge['target'].replace(df_vert['name'], inplace=True)
        
        network = ig.Graph.DataFrame(df_edge)
        return network
        
    def get_edgelist_df(self, network):
        df_vert = network.get_vertex_dataframe()
        df_edge = network.get_edge_dataframe().reset_index(drop=True)
        
        df_edge['source'].replace(df_vert['name'], inplace=True)
        df_edge['target'].replace(df_vert['name'], inplace=True)
        
        return df_edge
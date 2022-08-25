import random
import numpy as np
import pandas as pd
import igraph as ig
from numpy.random import uniform

class SignallingCascade:
    
    def create_signalling_cascades(self):
        edges_to_add = []
        sampled_egenes = []
        
        if self.verbose:
            print ("Sampling Signalling Cascades...", flush=True)
            
        if self.ppi.vcount() <= self.num_kinases:
            raise Exception("Number of Tfs is Larger than the GRN Provided.", flush=True)
        
        for nsize in self.cascade_sizes:
            sampled_nodes = self.sample_kinases(nsize)
            edge_df, sampled_egenes = self.map_cascade(sampled_nodes, sampled_egenes)
            edges_to_add.append(edge_df)
        
        edges_to_add = pd.concat(edges_to_add)
        self.kinases = self.reattach_kinase(edges_to_add)
    
    def sample_kinases(self, nsize):
        return self.modular_sampling(self.ppi, nsize)
    
    def map_cascade(self, sampled_nodes, sampled_egenes):
        tfs = [f'TF_{i}' for i in range(self.num_tfs)]
        network = self.ppi.induced_subgraph(sampled_nodes)
        egenes = [f'E-Gene_{i}' for i in range(self.num_egenes) if not f'E-Gene_{i}' in sampled_egenes]
        
        df_vert = network.get_vertex_dataframe()
        df_edge = network.get_edge_dataframe().reset_index(drop=True)
        
        # map kinases to E-Genes
        tfs = random.choices(tfs, k=self.ntfs_per_cascades)
        egenes = random.choices(egenes, k=len(sampled_nodes))
        
        # add kinase to tf edge
        df_vert['egenes_name'] = egenes
        df_edge['source_new_name'] = df_edge['source']
        df_edge['target_new_name'] = df_edge['target']
        
        kinases = random.choices(df_vert['name'].values, k=self.ntfs_per_cascades)
        new_sources = df_vert.loc[df_vert.name.isin(kinases), 'egenes_name'].values
        
        new_edges = pd.DataFrame({'source': kinases,
                                  'target': tfs,
                                  'source_new_name': new_sources,
                                  'target_new_name': tfs})
        
        sampled_egenes = sampled_egenes + egenes
        df_edge['source_new_name'] = df_edge['source']
        df_edge['target_new_name'] = df_edge['target']
        df_edge = pd.concat([df_edge, new_edges])
        
        df_edge['source'].replace(df_vert['name'], inplace=True)
        df_edge['target'].replace(df_vert['name'], inplace=True)
        df_edge['source_new_name'].replace(df_vert['egenes_name'], inplace=True)
        df_edge['target_new_name'].replace(df_vert['egenes_name'], inplace=True)
        
        return df_edge, sampled_egenes
    
    def reattach_kinase(self, df):
        new_regs = []
        cascade_net = ig.Graph.DataFrame(df)
        outdegree = cascade_net.degree(mode='out')
        tfs = [f'TF_{i}' for i in range(self.num_tfs)]
        
        for index, gene in enumerate(cascade_net.vs['name']):
            if (outdegree[index] == 0) and (not gene in tfs):
                new_regs.append(gene)
        
        if len(new_regs) > 0:
            reg_new_names = []
            target_new_names = []
            new_targets = random.choices(df.source.unique(), k=len(new_regs))
            
            for reg in new_regs:
                reg_new_names.append(df.loc[df.target == reg, 'target_new_name'].unique()[0])
            
            for target in new_targets:
                target_new_names.append(df.loc[df.source == target, 'source_new_name'].unique()[0])
            
            df = pd.concat([df, pd.DataFrame({'source': new_regs,
                                             'target': new_targets,
                                             'source_new_name': reg_new_names,
                                             'target_new_name': target_new_names})])
            
        return ig.Graph.DataFrame(df[['source_new_name', 'target_new_name']])
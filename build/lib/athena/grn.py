import random
import numpy as np
import pandas as pd
import igraph as ig
from numpy.random import uniform

class GeneRegulatoryNetwork:
    
    def create_grn(self):
        # load the GRN and extract the transcription factor network
        if self.verbose:
            print ("Cleaning Biological Network...", flush=True)
            
        self.grn = self.get_grn()
        tf_net = self.extract_tf_net()
        
        if tf_net.vcount() <= self.num_tfs:
            raise Exception("Number of Tfs is Larger than the GRN Provided...", flush=True)
        
        # sample tfs and e-genes
        if self.verbose:
            print ("Sampling TFs and E-Genes Network...", flush=True)
       
        tfs, tf_net = self.sample_tfs(tf_net)
        self.tf_egene_net = self.sample_egenes(tfs, tf_net)
    
    def get_grn(self):
        grn = pd.read_csv(self.grn_fp).set_index('row_names')
        edgelist = {'source':[], 'target':[], 'edge_weight':[]}

        for rowname, row in grn.iterrows():
            for colname in row.index:
                weight = row.loc[colname]

                if weight == 0:
                    continue

                edgelist['target'].append(colname)
                edgelist['source'].append(rowname)
                edgelist['edge_weight'].append(weight)

        edgelist = pd.DataFrame(edgelist)
        grn = ig.Graph.DataFrame(edgelist)
        return grn
    
    def extract_tf_net(self):
        tf_nodes = []
        nodes = [node.index for node in self.grn.vs]
        degrees = self.grn.degree(mode='out')

        for node_i, node in enumerate(nodes):
            if degrees[node_i] == 0:
                continue

            tf_nodes.append(node)

        tf_net = self.grn.induced_subgraph(tf_nodes)
        tf_net = self.clean_tf_net(tf_net)
        return tf_net
    
    def sample_tfs(self, tf_net):
        tfs = self.modular_sampling(tf_net, self.num_tfs)
        tf_net = tf_net.induced_subgraph(tfs)
        tf_net = self.rename_nodes(tf_net, tfs, 'TF')
        
        return tfs, tf_net
        
    def sample_egenes(self, tfs, tf_net):
        neighbors = self.get_neighbors(self.grn, tfs, 'out')
        tf_degree_net = self.grn.induced_subgraph(neighbors + tfs)
        tf_degree_net = self.rename_nodes(tf_degree_net, tfs, 'TF')
        df = self.get_edgelist_df(tf_degree_net)
        df = df.loc[df['source'].str.contains('TF_')]
        regs, probs, nrows = [], [], df.shape[0]
        
        for reg in df.source.unique():
            prob = df.loc[df.source == reg].shape[0] / nrows
            
            regs.append(reg)
            probs.append(prob)
        
        df = self.get_edgelist_df(tf_net)
        egenes = [f'E-Gene_{i}' for i in range(self.num_egenes)]
        new_regs = random.choices(regs, weights=probs, k=self.num_egenes)
        df = pd.concat([df, pd.DataFrame({'source': new_regs, 'target': egenes, 'edge_weight': uniform(size=self.num_egenes)})])
        return ig.Graph.DataFrame(df)
    
    def clean_tf_net(self, tf_net):
        graphs = tf_net.decompose(mode="WEAK")
        graph_edges = [graph.ecount() for graph in graphs]
        most_edges = max(graph_edges)

        for index, edge_count in enumerate(graph_edges):

            if edge_count == most_edges:
                tf_net = graphs[index]

        return tf_net
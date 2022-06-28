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
        self.extract_tf_net()
        
        if self.tf_net.vcount() <= self.num_tfs:
            raise Exception("Number of Tfs is Larger than the GRN Provided...", flush=True)
        
        # sample tfs and e-genes
        if self.verbose:
            print ("Sampling TFs and E-Genes Network...", flush=True)
       
        self.sample_tfs()
        self.sample_egenes()
        
        # clean sampled network
        if self.verbose:
            print ("Pruning invalid edges...", flush=True)
            
        network = self.network_qc()
        self.tf_egene_net = self.rename_nodes(network, self.tfs, 'TF')
        self.tf_egene_net = self.rename_nodes(self.tf_egene_net, self.egenes, 'E-Gene')
    
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
        self.tf_net = self.clean_tf_net(tf_net)
        
    def sample_tfs(self):
        self.tfs = self.modular_sampling(self.tf_net, self.num_tfs)
        
    def sample_egenes(self):
        temp = self.grn
        
        # get neighbors from GRN of transcription factors
        neighbors = self.get_neighbors(temp, self.tfs, 'out')
        temp = temp.induced_subgraph(neighbors + self.tfs)
        
        # use page rank to calculate node weights
        page_rank = temp.pagerank(directed=True, weights=temp.es['edge_weight'], damping=0.1)
        page_rank = page_rank + np.random.uniform(size=len(page_rank))
        
        # remove TFs from Page Rank Results
        page_df = pd.DataFrame({'feature_id': temp.vs['name'], 'weight': page_rank})
        page_df = page_df.loc[~page_df['feature_id'].isin(self.tfs), ]
        
        if len(neighbors) < self.num_egenes:
            num_egenes = len(neighbors)
        else:
            num_egenes = self.num_egenes
        
        if len(neighbors) != len(page_rank):
            neighbors = neighbors[:len(page_rank)]
        
        # sample E-Genes from the neighbors
        self.egenes = random.choices(page_df['feature_id'].values,
                                     weights=page_df['weight'].values,
                                     k=num_egenes)
    
    def network_qc(self):
        network = self.remove_egenes_to_tf_edges()
        network = self.reattach_egenes(network)
        return network
    
    def clean_tf_net(self, tf_net):
        graphs = tf_net.decompose(mode="WEAK")
        graph_edges = [graph.ecount() for graph in graphs]
        most_edges = max(graph_edges)

        for index, edge_count in enumerate(graph_edges):

            if edge_count == most_edges:
                tf_net = graphs[index]

        return tf_net
    
    def remove_egenes_to_tf_edges(self):
        sampled_genes = self.tfs + self.egenes
        temp = self.grn.induced_subgraph(sampled_genes)
        df = self.get_edgelist_df(temp)
            
        df['valid_edge'] = False
        df.loc[(df.source.isin(sampled_genes)) | (df.target.isin(sampled_genes)), 'valid_edge'] = True
        df.loc[(~df.source.isin(self.tfs)) & (df.target.isin(self.tfs)), 'valid_edge'] = False
        
        df = df.loc[df.valid_edge, ]
        df = df.drop(columns=['valid_edge'])
        
        grn = ig.Graph.DataFrame(df)
        return grn
    
    def reattach_egenes(self, network):
        df = self.get_edgelist_df(network)
        missing_targets = [egene for egene in self.egenes if not egene in df['target'].values]
        
        if len(missing_targets) > 0:
            prob = self.network.degree(mode='out')
            prob = np.array(prob) / sum(prob)
            new_regs = random.choices(self.network.vs['name'], weights=prob, k=len(missing_targets))
            df = pd.concat([df, pd.DataFrame({'source': new_regs, 'target': missing_targets,
                                              'edge_weight': uniform(size=len(missing_targets))})])
        
        return ig.Graph.DataFrame(df)
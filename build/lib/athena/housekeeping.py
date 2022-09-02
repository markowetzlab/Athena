import random
import numpy as np
import pandas as pd
import igraph as ig

class HouseKeeping:
    
    def create_hks(self):
        
        if self.verbose:
            print ("Sampling Housekeeping Network...")
        
        # select random node to serve as starting point
        sampled_nodes = self.breadth_first_search()
        hks = self.grn.induced_subgraph(sampled_nodes)
        
        # removing feedback loops
        hks = hks.get_edge_dataframe()
        hks = hks.loc[hks['source'] != hks['target'], ]
        
        self.hks = ig.Graph.DataFrame(hks)
        self.hks = self.rename_nodes(self.hks, self.hks.vs['name'], 'HK')
    
    def breadth_first_search(self):
        node_order =[]
        nodes = self.grn.vs['name']
        
        while len(node_order) <= self.num_hks:
            seed_node = random.sample(nodes, k=1)[0]
            seed_node = [i for i, n in enumerate(self.grn.vs['name']) if n == seed_node][0]
            
            node_order = self.grn.bfs(seed_node)[0]
            
        return node_order[:self.num_hks]
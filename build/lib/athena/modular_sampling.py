import random
import pandas as pd
import igraph as ig

class ModularSampling:
    """Responsible for sampling transcription factors from a GRN using modular sampling"""
    
    def modular_sampling(self, network, nsize, neighbor_mode='out'):
        sampled_nodes = []
        node_to_add = None
        nodes = [n['name'] for n in network.vs]
        modular_clusters = self.setup_modular_clusters(nodes)
        
        while len(sampled_nodes) < nsize:

            if len(sampled_nodes) == 0 or node_to_add is None:
                # randomly sample an initial node to start the sampling process
                node_to_add = random.sample(nodes, k=1)[0]
            else:
                node_to_add = self.highest_modular_node(network, sampled_nodes, modular_clusters, neighbor_mode)
            
            # if node_to_add is None: # may be causing an issue
                # sampled_nodes = []
            
            if (not node_to_add in sampled_nodes) and (not node_to_add is None):
                sampled_nodes.append(node_to_add)
                modular_clusters[node_to_add] = 1
        
        return sampled_nodes
        
    def highest_modular_node(self, network, sampled_nodes, modular_clusters, neighbor_mode):
        node_to_add = None
        largest_modularity = 0
        neighboring_nodes = self.get_neighbors(network, sampled_nodes, neighbor_mode)
        
        for node in neighboring_nodes:
            # setup membership with this node added
            if not node in modular_clusters.keys():
                continue
            
            clusters = modular_clusters
            clusters[node] = 1
            
            # calculate modularity with this nodes membership
            modularity = network.modularity(clusters.values())
            
            # if modularity is larger than current update variables
            if modularity >= largest_modularity:
                node_to_add = node
                largest_modularity = modularity
        
        return node_to_add

    def get_neighbors(self, network, nodes, neighbor_mode):
        neighboring_nodes = []
        
        for node in nodes:
            for neighbor in network.neighbors(node, mode=neighbor_mode):
                neighbor = network.vs['name'][neighbor]
                
                if (neighbor in nodes) or (neighbor in neighboring_nodes):
                    continue
                
                neighboring_nodes.append(neighbor)

        return neighboring_nodes
    
    def setup_modular_clusters(self, provided_nodes):
        modular_clusters = {}

        for node in provided_nodes:
            modular_clusters[node] = 0
            
        return modular_clusters
    
import os
import math
import random
import numpy as np
import pandas as pd
import pyopencl as cl
from tqdm import tqdm
from pyopencl import array as ocl_array
from multiprocessing import Pool, RLock

class GillespieSSA:
    
    def run(self):
        
        for batch_i in tqdm(range(self.nbatches), desc='Batch For Loop',leave=False):
            self.simulate(batch_i)
            
    def simulate(self, batch_number):
        # Setup Simulation Context and Parameters
        print ("Creating Command Queue...")
        context, program, cache_interval, not_perturbed, time_span = self.setup_simulation()
        
        with cl.CommandQueue(context) as queue:
            print ("Start Simulation...")
            aff_params, prop_params, change_params, species_array, species_vec, prop_df, affinities, propensities = self.initialize_variables(queue, batch_number)
            results, result_index = self.setup_results(species_vec)
            cache_species = species_vec.iloc[result_index]
            
            for time_index in tqdm(time_span, desc=f'Running Simulations Batch: {batch_number}'):
                # calculate affinites and propensities
                affinities = self.calc_affinities(program, queue, aff_params, species_array, affinities)
                propensity_vec = self.calc_propensities(program, queue, prop_params, propensities, affinities, species_array)

                # sample reactions and update species state
                k = self.get_firings(queue, propensity_vec)
                species_array = self.update_state(program, queue, species_array, results, k, change_params)

                # cache result and introduce perturbation to simulation
                not_perturbed, prop_params = self.perturb_reactions(queue, time_index, not_perturbed, prop_params, prop_df)
                cache_interval, results = self.cache_results(batch_number, time_index, cache_interval, cache_species, 
                                                             species_array, results, result_index)
            
            cache_interval, results = self.cache_results(batch_number, time_index, cache_interval, cache_species, 
                                                         species_array, results, result_index)
    
    def setup_simulation(self):
        cache_interval = 0
        not_perturbed = self.not_perturbed
        time_span = np.arange(0, self.sim_time, self.tau)
        
        context, program = self.get_gpu_context()
        
        return context, program, cache_interval, not_perturbed, time_span
    
    def setup_results(self, species_vec):
        if self.save_protein: # remove this option
            result_index = species_vec.index.values
        else:
            result_index = species_vec.loc[species_vec.molecule_type != 'protein', ].index.values
        
        result = np.zeros(shape=(self.cache_size, len(result_index)))
        return result, result_index
    
    def initialize_variables(self, queue, device_num):
        file = f'batch_{device_num}.parquet'
        aff_params, prop_params, change_params = {}, {}, {}
        affinity = pd.read_parquet(os.path.join(self.simulator_dir, 'affinity', file))
        regulators = pd.read_parquet(os.path.join(self.simulator_dir, 'regulators', file))
        propensity = pd.read_parquet(os.path.join(self.simulator_dir, 'propensity', file))
        change_vec = pd.read_parquet(os.path.join(self.simulator_dir, 'change_vec', file))
        species_vec = pd.read_parquet(os.path.join(self.simulator_dir, 'species_vec', file))
        
        # stable affinities parameters
        aff_params['hill'] = ocl_array.to_device(queue, affinity['hill'].values.astype(np.float64))
        aff_params['strengths'] = ocl_array.to_device(queue, affinity['strength'].values.astype(np.float64))
        aff_params['regulatory_index'] = ocl_array.to_device(queue, affinity['from_index'].values.astype(np.int32))
        aff_params['dissociations'] = ocl_array.to_device(queue, affinity['dissociation'].values.astype(np.float64))
        
        # stable propensities parameters
        prop_params['effects'] = ocl_array.to_device(queue, affinity['effect'].values.astype(np.int32))
        prop_params['basal'] = ocl_array.to_device(queue, propensity['basal'].values.astype(np.float64))
        prop_params['regulators'] = ocl_array.to_device(queue, regulators['regulators'].values.astype(np.int32))
        prop_params['nregulators'] = ocl_array.to_device(queue, propensity['nregulators'].values.astype(np.int32))
        prop_params['effects_sums'] = ocl_array.to_device(queue, propensity['effects_sums'].values.astype(np.int32))
        prop_params['independence'] = ocl_array.to_device(queue, propensity['independence'].values.astype(np.float64))
        prop_params['reaction_type'] = ocl_array.to_device(queue, propensity['reaction_type'].values.astype(np.int32))
        prop_params['species_index'] = ocl_array.to_device(queue, propensity['species_index'].values.astype(np.int32))
        prop_params['base_activity'] = ocl_array.to_device(queue, propensity['base_activity'].values.astype(np.float64))
        prop_params['reaction_rates'] = ocl_array.to_device(queue, propensity['reaction_rate'].values.astype(np.float64))
        prop_params['reversible_reactions'] = ocl_array.to_device(queue, propensity['reversible_reaction'].values.astype(np.int32))
        
        # stable update state parameters
        change_params['reaction_effects'] = ocl_array.to_device(queue, change_vec['reaction_effect'].values.astype(np.int32))
        change_params['change_reaction_index'] = ocl_array.to_device(queue, change_vec['react_index'].values.astype(np.int32))
        change_params['change_species_index'] = ocl_array.to_device(queue, change_vec['species_index'].values.astype(np.int32))
        
        # create species, affinities and propensities vector
        species_array = ocl_array.to_device(queue, species_vec['state'].values.astype(np.int32))
        propensities = ocl_array.zeros(queue, order='C', shape=prop_params['basal'].shape, dtype=np.float64)
        affinities = ocl_array.zeros(queue, order='C', shape=aff_params['strengths'].shape, dtype=np.float64)
        return aff_params, prop_params, change_params, species_array, species_vec, propensity, affinities, propensities
    
    def calc_affinities(self, program, queue, aff_params, species_array, affinities):
        # initialize affinities parameters
        affinities_calc = program.calc_affinities(queue,
                                                   aff_params['strengths'].shape,
                                                   None,
                                                   aff_params['strengths'].data,
                                                   aff_params['hill'].data,
                                                   aff_params['dissociations'].data,
                                                   aff_params['regulatory_index'].data,
                                                   species_array.data,
                                                   affinities.data)

        affinities_calc.wait()
        return affinities

    def calc_propensities(self, program, queue, prop_params, propensities, affinities, species_array):
        # initializing propensity variables
        propensities_calc = program.calc_propensities(queue,
                                                        prop_params['reaction_rates'].shape,
                                                        None,
                                                        prop_params['effects'].data,
                                                        prop_params['regulators'].data,
                                                        prop_params['effects_sums'].data,
                                                        prop_params['nregulators'].data,
                                                        species_array.data,
                                                        prop_params['species_index'].data,
                                                        prop_params['reaction_type'].data,
                                                        prop_params['reversible_reactions'].data,
                                                        prop_params['basal'].data,
                                                        affinities.data,
                                                        propensities.data,
                                                        prop_params['independence'].data,
                                                        prop_params['base_activity'].data,
                                                        prop_params['reaction_rates'].data)

        propensities_calc.wait()
        return propensities.get()
    
    def get_firings(self, queue, propensity_vec):
        # print (max(propensity_vec))
        k = np.random.poisson(propensity_vec * self.tau).astype(np.int32)
        return ocl_array.to_device(queue, k)
    
    def update_state(self, program, queue, species_array, results, k, change_params):
        # initialize update species data parameters
        state_calc = program.update_state(queue,
                                           change_params['reaction_effects'].shape,
                                           None,
                                           species_array.data,
                                           change_params['change_species_index'].data,
                                           change_params['change_reaction_index'].data,
                                           change_params['reaction_effects'].data,
                                           k.data)
        
        state_calc.wait()
        return species_array
    
    def perturb_reactions(self, queue, time_index, not_perturbed, prop_params, prop_df):
        
        if time_index >= self.perturb_time and not_perturbed:
            not_perturbed = False
            perturb_rates = prop_df['reaction_rate'] * prop_df['perturbation']
            prop_params['reaction_rates'] = ocl_array.to_device(queue, perturb_rates.values.astype(np.float64))
        
        return not_perturbed, prop_params
    
    def cache_results(self, device_n, time_index, cache_interval, cache_species, species_array, results, result_index):
        
        if (time_index >= self.sample_time) and (time_index % self.update_interval == 0):
            results[cache_interval] = species_array.get()[result_index]
            
            if cache_interval == (results.shape[0] - 1):
                sims = cache_species.sim_i.unique()
                
                for sim in sims:
                    spec = cache_species.loc[cache_species.sim_i == sim,]
                    sim_res = results[:, spec.index]
                    colnames = list(spec.spec_name.values)
                    fp = os.path.join(self.results_dir, f'simulated_counts.csv.gz')
                    meta_fp = os.path.join(self.results_dir, f'cell_metadata.csv.gz')
                    
                    df = pd.DataFrame(sim_res, columns=colnames)
                    
                    if self.collapse_mrna:
                        spec = spec.sort_values(by=['gene'])
                        mrna = spec.loc[spec.molecule_type == 'mrna', 'spec_name'].values
                        premrna = spec.loc[spec.molecule_type == 'premrna', 'spec_name'].values
                        df = pd.DataFrame(df[mrna].values + df[premrna].values, columns=mrna)
                    
                    df = df.apply(pd.to_numeric)
                    df = self.manage_dtypes(df)
                    cell_meta = pd.DataFrame({'sim_i': [sim] * df.shape[0]})
                    
                    if os.path.exists(fp):
                        df.to_csv(fp, mode='a', index=False, header=False, compression='gzip')
                        cell_meta.to_csv(meta_fp, mode='a', index=False, header=False, compression='gzip')
                    else:
                        df.to_csv(fp, index=False, compression='gzip')
                        cell_meta.to_csv(meta_fp, index=False, compression='gzip')
                
                results = np.zeros(shape=(self.cache_size, len(result_index)))    
                cache_interval = 0
            else:
                cache_interval += 1
        
        return cache_interval, results
    
    def get_gpu_context(self):
        context = cl.create_some_context(True)
        
        with open(os.path.join(os.path.dirname(__file__), 'utils.cl'), 'r') as file:
            code = file.read()
            
        program = cl.Program(context, code).build(
                            options=['-cl-denorms-are-zero',
                                     '-cl-no-signed-zeros',
                                     '-cl-finite-math-only',
                                     '-cl-mad-enable'])
        return context, program

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
    #error "Double doubleision doubleing point not supported by OpenCL implementation."
#endif


__kernel void calc_affinities(__global double* strength,
                              __global double* hill,
                              __global double* dissociation,
                              __global int* regulatory_index,
                              __global int* species_state,
                              __global double* affinities) {
    // get thread id
    const int tid =  get_global_id(0);
    
    // calculate affinity
    affinities[tid] = strength[tid] * pow(species_state[regulatory_index[tid]] / dissociation[tid], hill[tid]);
}

__kernel void calc_propensities(__global int* effects,
                                __global int* regulators,
                                __global int* effect_sums,
                                __global int* nregulators,
                                __global int* species_state,
                                __global int* species_index,
                                __global int* reaction_type,
                                __global int* reversible_reaction,
                                __global double* basal,
                                __global double* affinities,
                                __global double* propensities, 
                                __global double* independence,
                                __global double* base_activity,
                                __global double* reaction_rates) {
        
        const int tid = get_global_id(0);  

        if (reaction_type[tid] == 0) {
            // calculate propensity function for constant reactions
            propensities[tid] = reaction_rates[tid] * species_state[species_index[tid]];
        } else if (reaction_type[tid] == 1 && effect_sums[tid] == 0) {
            // basal expression propensity
            propensities[tid] = reaction_rates[tid] * basal[tid];
        } else if (reaction_type[tid] == 1 && nregulators[tid] != 0 ){
            // arbritary N regulatory or protein interactions propensity function
            double numerator = 1.0;
            double denominator = 1.0;

            for (int i=0; i < nregulators[tid]; i++) {
                int rindex = regulators[tid + i];

                // if regulatory is a promoter of reaction add to numerator of propensity func
                if (effects[rindex] > 0) {
                    numerator *= (affinities[rindex] + independence[tid]);
                }

                denominator *= (affinities[rindex] + 1);
            }
            
            // calculate propensity
            numerator = base_activity[tid] + numerator;
            propensities[tid] = reaction_rates[tid] * (numerator / denominator);

            if (reversible_reaction[tid] == 1) {
                // if reverse reaction calculate inverse of forward propensity
                propensities[tid] = 1 / propensities[tid];
            }
        }
}


__kernel void update_state(__global int* species_state,
                           __global int* species_index,
                           __global int* reaction_index,
                           __global int* reaction_effect,
                           __global int* k) {
    
    const int tid = get_global_id(0);
    species_state[species_index[tid]] += (reaction_effect[tid] * k[reaction_index[tid]]);
    
    if (species_state[species_index[tid]] < 0) {
        species_state[species_index[tid]] = 0;
    }
}
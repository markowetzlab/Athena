import os
import glob
import pickle
import random
import shutil
import zipfile
import requests
import igraph as ig
import pandas as pd
from .grna import GuideRNA
from .network import Network
from .kinetics import Kinetics
from .sampling import Sampling
from .gillespie_ssa import GillespieSSA
from .compile_reactions import CompileReactions

class Athena(Network, Kinetics, GuideRNA, CompileReactions, GillespieSSA, Sampling):
    grn_dir = 'GRNs'
    grna_libraries_dir = 'gRNA_Libraries'
    sc_pops_dir = 'Single_Cell_Populations'
    ppi_fp = 'Signalling_Pathways/reactome_human_ppi.csv'
    url = 'https://zenodo.org/api/files/0a6d457c-81e1-4296-8254-7f312b0ef77d/athena.zip'
                
    def __init__(self,
                 ncpus=1,
                 ntfs=10,
                 nhks=100,
                 negenes=100,
                 nkinases=10,
                 ncascades=1,
                 ntfs_per_cascades=1,
                 feature_info=None,
                 feature_network=None,
                 grn_net=None,
                 signalling_net=None,
                 target_genes=None,
                 perturb_tfs=True,
                 perturb_kinases=True,
                 on_target=None,
                 off_target=None,
                 ctrl_label='CTRL',
                 crispr_type=None,
                 ngrnas_per_target=3,
                 grna_library=None,
                 tau=0.05,
                 nbatches=20,
                 noise_mean=1.0, 
                 noise_sd=0.005,
                 burn_time=100,
                 perturb_time=100,
                 sample_time=100,
                 update_interval=1,
                 cache_interval=50,
                 save_burn=False,
                 save_protein=False,
                 ncells_per_condition=10000,
                 collapse_mrna=True,
                 map_reference_ls=True,
                 map_reference_cpm=True,
                 simulator_dir='athena',
                 opencl_root='/opt/rocm/opencl',
                 opencl_context="0",
                 cache_dir=os.path.join(os.environ['HOME'], '.cache/athena'),
                 cache_network=True,
                 verbose=False):
        
        print ("Initiate Environmental Parameters...")
        self.ncpus = ncpus
        self.verbose = verbose
        self.cache_network = cache_network
        self.network_name = os.path.basename(simulator_dir)
        self.initiate_opencl(nbatches, opencl_context, opencl_root)
        
        print ("Check the caches...")
        self.check_caches(cache_dir)
        
        print ("Setup Simulator Directory...")
        self.setup_simulator_directory(simulator_dir)
        
        print ("Check Network Parameters...")
        self.check_network_parameters(ntfs, nhks, negenes, nkinases, ncascades, ntfs_per_cascades, feature_info, feature_network, grn_net, signalling_net)
        
        print ("Check gRNA Parameters...")
        self.check_grna_parameters(target_genes, perturb_tfs, perturb_kinases, on_target, off_target, ngrnas_per_target, ctrl_label, crispr_type, grna_library)
        
        print ("Check Simulation Parameters...")
        self.check_simulation_parameters(tau, noise_mean, noise_sd, burn_time, perturb_time, sample_time,
                                         update_interval, cache_interval, save_burn, save_protein, ncells_per_condition)
        
        print ("Check Downsampling Parameters...")
        self.check_downsample_parameters(collapse_mrna, map_reference_ls, map_reference_cpm)
    
    def cache(self, fp=None):
        
        if fp is None:
            fp = 'athena.pkl'
        
        with open(fp, 'wb') as file:
            pickle.dump(self, file)
    
    def initiate_opencl(self, nbatches, context, opencl_root):
        self.nbatches = nbatches
        self.opencl_root = opencl_root
        
        # setting up opencl environmental variables
        os.environ['PYOPENCL_CTX'] = context
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['OPENCL_ROOT'] = opencl_root
        
    def setup_simulator_directory(self, simulator_dir):
        self.simulator_dir = simulator_dir
        
        self.results_dir = os.path.join(self.simulator_dir, 'results')
        self.metadata_dir = os.path.join(self.simulator_dir, 'metadata')
        self.affinity_dir = os.path.join(self.simulator_dir, 'affinity')
        self.multiplier_dir = os.path.join(self.simulator_dir, 'multiplier')
        self.propensity_dir = os.path.join(self.simulator_dir, 'propensity')
        self.change_vec_dir = os.path.join(self.simulator_dir, 'change_vec')
        self.regulators_dir = os.path.join(self.simulator_dir, 'regulators')
        self.species_vec_dir = os.path.join(self.simulator_dir, 'species_vec')
        
        dirs_to_create = [self.simulator_dir, self.results_dir, self.metadata_dir,
                          self.affinity_dir, self.multiplier_dir, self.propensity_dir,
                          self.change_vec_dir, self.regulators_dir, self.species_vec_dir]
        
        for dir_to_create in dirs_to_create:
            try:
                os.mkdir(dir_to_create)
            except:    
                
                for file in os.listdir(dir_to_create):
                    fp = os.path.join(dir_to_create, file)
                    os.remove(fp)
                    
    def check_caches(self, cache_dir):
        self.cache_dir = cache_dir
        self.grn_dir = os.path.join(cache_dir, self.grn_dir)
        self.sc_pops_dir = os.path.join(cache_dir, self.sc_pops_dir) 
        self.grna_libraries_dir = os.path.join(cache_dir, self.grna_libraries_dir)
        metadata_exists = os.path.isdir(self.grn_dir) or os.path.isdir(self.sc_pops_dir) or os.path.isdir(self.grna_libraries_dir)
        
        if not metadata_exists:
            self.download_metadata()
    
    def check_network_parameters(self, ntfs, nhks, negenes, nkinases, ncascades, ntfs_per_cascades, feature_info, feature_network, grn_net, pathway):
        """ Check the Network Parameters and make sure that all of the parameters are okay. """
        
        if feature_info is None or feature_network is None:
            self.feature_info = None
            self.feature_network = None
            grns = os.listdir(self.grn_dir)

            if not type(ntfs) is int:
                raise Exception('ntfs: Parameter Must be Integer Data Type...')

            if not type(nhks) is int:
                raise Exception('nhks: Parameter Must be Integer Data Type...')

            if not type(negenes) is int:
                raise Exception('negenes: Parameter Must be Integer Data Type...')

            if not type(nkinases) is int:
                raise Exception('nkinases: Parameter Must be Integer Data Type...')
                
            if nkinases > (negenes / 2):
                raise Exception('nkinases: Must be half of Egenes...')
            
            if not type(ntfs_per_cascades) is int:
                raise Exception('ntfs_per_cascades: Parameter Must be Integer Data Type...')
                
            if (grn_net is None) or (not grn_net in grns):
                grn = random.sample(os.listdir(self.grn_dir), k=1)[0]
                self.grn_fp = os.path.join(self.grn_dir, grn)
            else:
                self.grn_fp = os.path.join(self.grn_dir, grn_net)

            # Comment this out  until signalling pathway has been implemented.
            ppi = pd.read_csv(os.path.join(self.cache_dir, self.ppi_fp))
            ppi = ppi.loc[ppi.Interactor_1_uniprot_id != ppi.Interactor_2_uniprot_id, ]
            self.ppi = ig.Graph.DataFrame(ppi)
            self.cascade_sizes = self.calc_cascade_sizes(nkinases, ncascades)
            
            self.ntfs_per_cascades = ntfs_per_cascades
            self.num_kinases, self.num_cascades = nkinases, ncascades
            self.num_tfs, self.num_hks, self.num_egenes, self.nkinases = ntfs, nhks, negenes, nkinases
            
        else:
            self.feature_info = pd.read_csv(feature_info)
            self.feature_network = pd.read_csv(feature_network)
            self.feature_info.to_csv(os.path.join(self.metadata_dir, 'feature_info.csv.gz'), compression='gzip')
            self.feature_network.to_csv(os.path.join(self.metadata_dir, 'feature_network.csv.gz'), compression='gzip')
    
    def check_grna_parameters(self, target_genes, perturb_tfs, perturb_kinases, on_target, off_target, ngrnas_per_target, ctrl_label, crispr_type, grna_library):
        genes = []
        self.target_genes = []
        self.on_target, self.off_target = None, None
        self.perturb_tfs, self.perturb_kinases = False, False
        grna_libraries = os.listdir(self.grna_libraries_dir)
        
        try:
            crispr_type = crispr_type.lower()
            if crispr_type in ['interference', 'activation', 'knockout']:
                self.crispr_type = crispr_type
            else:
                raise Exception("crispr_type parameter must be either: activation, interference, or knockout...")
        except:
            self.crispr_type = None
        
        if grna_library in grna_libraries:
            self.grna_library_name = grna_library
            lib_fp = os.path.join(self.grna_libraries_dir, self.grna_library_name)
        else:
            self.grna_library_name = random.sample(grna_libraries, k=1)[0]
            lib_fp = os.path.join(self.grna_libraries_dir, self.grna_library_name)
        
        self.grna_library = pd.read_csv(lib_fp).drop(columns=['Unnamed: 0'])
        
        if (target_genes is None) and perturb_tfs:
            self.perturb_tfs = perturb_tfs
                
        if (target_genes is None) and perturb_kinases:
            self.perturb_kinases = perturb_kinases
        
        if (type(target_genes) is list) and (len(target_genes) != 0):
            self.target_genes = target_genes
        
        if (not on_target is None) and (type(on_target) is float) and (on_target <= 1) and (on_target >= 0):
            self.on_target = on_target
        
        if (not off_target is None) and (type(off_target) is float) and (off_target <= 1) and (off_target >= 0):
            self.off_target = off_target
        
        if ngrnas_per_target < 0:
            ngrnas_per_target = 1
        
        self.target_genes = genes
        self.ctrl_label = ctrl_label
        self.ngrnas_per_target = ngrnas_per_target
        
    def check_simulation_parameters(self, tau, noise_mean, noise_sd, burn_time, perturb_time, sample_time,
                                    update_interval, cache_interval, save_burn, save_protein, ncells_per_condition):
        
        if not type(tau) is float and tau <= 1.0:
            raise Exception('tau: Parameter Must be Float Data Type and less than 1...')
        
        if not type(noise_mean) is float:
            raise Exception('noise_mean: Parameter Must be Float Data Type...')
        
        if not type(noise_sd) is float:
            raise Exception('noise_sd: Parameter Must be Float Data Type...')
            
        if not type(burn_time) is int:
            raise Exception('burn_time: Parameter Must be Integer Data Type...')
        
        if not type(perturb_time) is int:
            raise Exception('perturb_time: Parameter Must be Integer Data Type...')
        
        if not type(sample_time) is int:
            raise Exception('sample_time: Parameter Must be Integer Data Type...')
        
        if not type(update_interval) is int:
            raise Exception('update_interval: Parameter Must be Integer Data Type...')
            
        if not type(cache_interval) is int:
            raise Exception('cache_interval: Parameter Must be Integer Data Type...')
        
        if not type(save_burn) is bool:
            raise Exception('save_burn: Parameter Must be Boolean Data Type...')
        
        if not type(save_protein) is bool:
            raise Exception('save_protein: Parameter Must be Boolean Data Type...')
            
        if perturb_time <= update_interval:
            raise Exception('perturb_time parameter must be greater than update_interval parameter...')
        
        self.tau = tau
        self.not_perturbed = True
        self.noise_std = noise_sd
        self.noise_mean = noise_mean
        self.cache_size = cache_interval
        self.save_protein = save_protein
        self.cache_interval = cache_interval
        self.update_interval = update_interval
        self.nsims_per_condition = int(ncells_per_condition / (sample_time / update_interval))
        
        # setting simulation time
        self.perturb_time = burn_time
        self.sample_time = burn_time + perturb_time
        self.sim_time = burn_time + perturb_time + sample_time
        
    def check_downsample_parameters(self, collapse_mrna, map_reference_ls, map_reference_cpm):
        
        if not type(collapse_mrna) is bool:
            raise Exception('collapse_mrna: Parameter Must be Boolean Data Type...')
        
        if not type(map_reference_ls) is bool:
            raise Exception('map_reference_ls: Parameter Must be Boolean Data Type...')
        
        if not type(map_reference_cpm) is bool:
            raise Exception('map_reference_cpm: Parameter Must be Boolean Data Type...')
        
        self.collapse_mrna = collapse_mrna
        self.map_reference_ls = map_reference_ls
        self.map_reference_cpm = map_reference_cpm
        
        pop_files = os.listdir(self.sc_pops_dir)
        pop_file = random.choices(pop_files)[0]
        self.pop_fp = os.path.join(self.sc_pops_dir, pop_file)
        
    def download_metadata(self):
        """Download the Required Metadata..."""
        print ('Downloading metadata...')
        # download zenodo metadata files
        zip_filename = os.path.basename(self.cache_dir) + '.zip'
        cache_base_dir = os.path.dirname(self.cache_dir)
        zip_filename = os.path.join(cache_base_dir, zip_filename)
        response = requests.get(self.url, allow_redirects=True)
        open(zip_filename, 'wb').write(response.content)

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
            filepaths = glob.glob(os.path.join(self.cache_dir,'**','*'), recursive=True)    

            for old_filepath in filepaths:

                if os.path.isdir(old_filepath):
                    continue

                filename = os.path.basename(old_filepath)   
                file_dir = os.path.basename(os.path.dirname(old_filepath))  

                dir_fp = os.path.join(self.cache_dir, file_dir)
                new_filepath = os.path.join(dir_fp , filename)

                if not os.path.exists(dir_fp):
                    os.mkdir(dir_fp)
                
                shutil.move(old_filepath, new_filepath)

            shutil.rmtree(os.path.join(self.cache_dir, 'athena'))
            shutil.rmtree(os.path.join(self.cache_dir, '__MACOSX'))

        os.remove(zip_filename)
                
    def calc_cascade_sizes(self, num_kinases, num_cascades):
        cascade_sizes = []
        
        while sum(cascade_sizes) < num_kinases:
            cascade_sizes = []
            kinase_count = num_kinases

            for cascade_i in range(num_cascades):
                
                if kinase_count == 0:
                    cascade_sizes = []
                
                try:
                    nk = random.sample(range(kinase_count), k=1)[0] + 1
                    kinase_count = kinase_count - nk
                    cascade_sizes.append(nk)
                except:
                    break
        
        return cascade_sizes
    
    def manage_dtypes(self, df):
        df = df.convert_dtypes()
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns
        
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
        df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
        return df
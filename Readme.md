# Athena

Athena is Single Cell CRISRP Simulator capable of simulating CRISPR Interference, Activation, and Knockout.

## Install Instructions for Athena

### Installing Pyopencl

If you have already installed Opencl on your local system then you'll need to install pyopencl. pyopencl requires a serious of other non-python depenencies to be installed and compiled. This can be done manually but I would highly recommend simply installing pyopencl via conda. Use the following commands to install Athena:

    conda install -c conda-forge pyopencl

If Opencl has been installed but is not accessible to the conda enviroment use:

    conda install -c conda-forge pyopencl ocl-idc-system

### Installing Opencl and Pyopencl

Opencl is crucial to the functionality of this software package. How you install opencl will depend upon the hardware your using. If your planning on using a GPU you'll need to refer to [Nvadia](https://developer.nvidia.com/opencl) and [AMDs](https://www.amd.com/en/support/kb/faq/amdgpu-installation#faq-Using-the-amdgpu-install-Script) opencl instruction pages. If you wish to simply use an Intel core processor then run the following command:

    conda install -c conda-forge pyopencl pocl

### Installing the Package

Once Opencl and pyopencl are both installed can you install athena using pip:

    pip install athenasc

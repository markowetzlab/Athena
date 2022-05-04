from setuptools import setup, find_packages

setup(name='athenasc',
      version='0.524',
      description='Athena - Single Cell Crispr Simulator for experimental and analysis pipeline design',
      author='Alexander Baker',
      author_email='alexander.baker@cruk.cam.ac.uk',
      packages=find_packages(),
      package_data={'athena':['utils.cl']},
      install_requires=['numba', 'scipy', 'pandas', 'igraph', 'tqdm', 'loompy', 'scanpy', 'requests', 'pyarrow', 'fastparquet'],
      tests_require=['numba', 'scipy', 'pandas', 'igraph', 'tqdm', 'loompy', 'scanpy', 'requests', 'pyarrow', 'fastparquet'],
      keywords=['systems', 'biology', 'model'],
      classifiers=[ 'Development Status :: 5 - Production/Stable',
                    'Environment :: Console',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Operating System :: OS Independent',
                    'Programming Language :: Python :: 3',
                    'Topic :: Scientific/Engineering :: Bio-Informatics'])

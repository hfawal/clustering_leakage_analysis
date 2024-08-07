from setuptools import setup

setup(
    name='leakyblobs',
    version='0.1.0',    
    description='Clustering leakage analysis library.',
    url='https://git.equancy.cloud/equancy/data-intelligence/clustering_leakage_analysis/-/tree/1_initial_dev_HF',
    author='Hady Fawal',
    author_email='hfawal@equancy.com',
    license='Equancy All Rights Reserved',
    packages=['leakyblobs'],
    install_requires=['numpy>=1.26.4', 
                      'pandas>=2.2.1',
                      'openpyxl>=3.1.5',
                      'pyvis>=0.3.2',
                      'plotly>=5.20.0',
                      'scipy>=1.14.0',
                      'openpyxl>=3.1.5'                   
                      ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',     
        'Programming Language :: Python :: 3.11',
    ],
)

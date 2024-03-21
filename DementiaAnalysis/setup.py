from setuptools import setup, find_packages

setup(
    name='DementiaAnalysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'nilearn',
        'itertools',
        'statsmodels',
    ],
)
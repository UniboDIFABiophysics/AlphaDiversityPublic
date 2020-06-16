#!/home/PERSONALE/claudia.sala3/SHARED_FOLDER/miniconda3/bin python
"""
## NOTE
perchè ho il valore exp_1 diverso fra i vari run??
    
"""
import scipy.stats as st
import numpy as np
import pandas as pd
# import pylab as plt
# from scipy.special import gamma
from collections import OrderedDict
from datetime import datetime
import sys
# import matplotlib as mpl
# import os
# from multiprocessing import Pool
# import seaborn
# from operator import mul
# import functools

# %%
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# %%
seed = int(config['RANDOM SEED']['seed'])
np.random.seed(seed=seed)

size = int(config['GENERAL']['size'])
max_power = int(config['GENERAL']['max_power'])
destination = config['GENERAL']['random_values_destination']
priors_dataset = config['GENERAL']['priors']
exp_i = float(config['GENERAL']['exponent'])

# %%


# %%
parametri = pd.read_csv(priors_dataset, index_col='Unnamed: 0')

mu1_dist =st.gamma(
    parametri.loc['a','mu1_t'], 
    scale= parametri.loc['b','mu1_t'],
    ).rvs
var1_dist =st.gamma(
    parametri.loc['a','var1_t'],
    scale= parametri.loc['b','var1_t'],
    ).rvs
mu2_dist =st.gamma(
    parametri.loc['a','mu2_t'], 
    scale= parametri.loc['b','mu2_t'],
    ).rvs
var2_dist =st.gamma(
    parametri.loc['a','var2_t'],
    scale= parametri.loc['b','var2_t'],
    ).rvs
a_dist=st.beta(
    parametri.loc['a','a_t'], 
    parametri.loc['b','a_t'],
    ).rvs

# %%
"""
Definizioni
"""
def generate_cdf_log2(n_t, p_t, max_power):
    idx = np.r_[0, 2**np.arange(max_power)]
    logcdf = st.nbinom.logcdf(idx, n_t, p_t)
    logcdf = logcdf-logcdf[0]
    cdf  = np.exp(logcdf)
    cdf = cdf[:-1]-cdf[1:]
    cdf = cdf/np.sum(cdf)
    cdf = np.cumsum(cdf)
    return cdf



def genera_parameter_distribution(S:int, max_power:int, exp_power:float):
    mu1_t = mu1_dist(size=S)
    var1_t = mu1_t + var1_dist(size = S)**exp_power ### solo primo round con **2
    p1_t = mu1_t/var1_t
    n1_t = (mu1_t**2)/(var1_t-mu1_t)

    mu2_t = mu1_t + mu2_dist(size=S)
    var2_t = mu2_t + var2_dist(size = S)**exp_power ### solo primo round con **2
    p2_t = mu2_t/var2_t
    n2_t = (mu2_t**2)/(var2_t-mu2_t)
    a_t = a_dist(size=S)

    cdfs_1 = np.empty((S, max_power))
    for idx, (n, p) in enumerate(zip(n1_t, p1_t)):
        cdfs_1[idx] = generate_cdf_log2(n, p, max_power = max_power)

    cdfs_2 = np.empty((S, max_power))
    for idx, (n, p) in enumerate(zip(n2_t, p2_t)):
        cdfs_2[idx] = generate_cdf_log2(n, p, max_power = max_power)

    database_random = OrderedDict()
    database_random['mu1_t'] = mu1_t
    database_random['var1_t'] = var1_t
    database_random['n1_t'] = n1_t
    database_random['p1_t'] = p1_t
    database_random['mu2_t'] = mu2_t
    database_random['var2_t'] = var2_t
    database_random['n2_t'] = n2_t
    database_random['p2_t'] = p2_t
    database_random['a_t'] = a_t
    for idx in range(max_power):
        database_random['CDF_1_idx_{}'.format(idx)] = cdfs_1[:, idx]
    for idx in range(max_power):
        database_random['CDF_2_idx_{}'.format(idx)] = cdfs_2[:, idx]
    database_random = pd.DataFrame(database_random)

    return database_random




# %%
#############################
# Simulate parameters: (2 sets of 5000000 parameters each; power=20 means at most 2^20 individuals per species)
#############################


database_random = genera_parameter_distribution(
    S=size, 
    max_power=max_power,
    exp_power=exp_i,
    )

# %%

database_random.to_csv(destination, sep='\t', index=False)

print("Saved as: ", destination, file=sys.stderr)

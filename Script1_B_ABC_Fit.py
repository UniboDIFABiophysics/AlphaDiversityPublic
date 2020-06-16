#!/home/PERSONALE/claudia.sala3/SHARED_FOLDER/miniconda3/bin python
import scipy.stats as st
import numpy as np
import pandas as pd
import pylab as plt
from scipy.special import gamma
from collections import OrderedDict
from datetime import datetime

import os
from multiprocessing import Pool

from operator import mul
import functools
import glob

############################
#
# PARAMETERS
#
############################

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# %%
seed = int(config['RANDOM SEED']['seed'])
np.random.seed(seed=seed)

random_values = config['GENERAL']['random_values_destination']

pool_size = int(config['ABC']['pool_size'])
# %%

############################
# Load otu table:
# Each column corresponds to one subject; each row to one otu.
# The first column contains otu labels and is named "#OTU ID"
# The file is tab separated
############################

# base_dir = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/"

# filename = base_dir+"data/OTUs_UPARSE/pooled_file_Biagisamples.otutab.txt"


filename = config['ABC']['data']


output_folder = config['ABC']['output_folder']

# %%
# ############################
# # Choose round and folders:
# ############################
# round_i = 2 # round_i = 1 or 2
# output_folder1 = base_dir+"Risultati_ABC_prova/ABC_round1/"
# output_folder2 = base_dir+"Risultati_ABC_prova/ABC_round2/"
# priors_folder1 = base_dir+"Risultati_ABC_prova/ABC_priors_round1/" # folder with priors for round 1 (and parameter estimates)
# priors_folder2 = base_dir+"Risultati_ABC_prova/ABC_priors_round2/" # folder with priors for round 2


# ############################
# # PRIOR PROBABILITY
# ############################

# if round_i == 1: # For round 1, the priors are always these
#     output_folder = output_folder1
#     mu1_dist = st.invgamma(1.5,1.).rvs
#     var1_dist = st.invgamma(1.5,1.).rvs
#     mu2_dist = st.invgamma(1.5,1.).rvs
#     var2_dist = st.invgamma(1.5,1.).rvs
#     a_dist = st.uniform(0.,1.).rvs

# if round_i == 2: # For round 2 load prior parameters previously computed from round 1
#     output_folder = output_folder2
#     filename = priors_folder1+'parametri.csv'
#     parametri = pd.read_csv(filename,index_col='Unnamed: 0')
#     mu1_dist =st.gamma(parametri.loc['a','mu1_t'] ,scale= parametri.loc['b','mu1_t'] ).rvs
#     var1_dist =st.gamma( parametri.loc['a','var1_t'] ,scale= parametri.loc['b','var1_t']  ).rvs
#     mu2_dist =st.gamma( parametri.loc['a','mu2_t'] ,scale= parametri.loc['b','mu2_t'] ).rvs
#     var2_dist =st.gamma( parametri.loc['a','var2_t'] ,scale= parametri.loc['b','var2_t']  ).rvs
#     a_dist=st.beta( parametri.loc['a','a_t'], parametri.loc['b','a_t'] ).rvs



# if round_i == 1:
#     exp_i = 2
#     priors_folder = priors_folder1
# if round_i == 2:
#     exp_i = 1
#     priors_folder = priors_folder2
    
    
# if round_i == 2:
#     print(parametri)
# %%
#############################################################################
# USE THE GENERATED TABLES FOR RANDOM ESTIMATION
#############################################################################

def generate_hists(cdf, N, max_power):
    estratti = np.searchsorted(cdf, plt.rand(N))
    count = np.bincount(estratti, minlength=max_power)
    return count

def read_parameter_distribution(filenames):
    
    databases = [pd.read_csv(name, sep='\t') for name in glob.glob(filenames)]
    for filename in glob.glob(filenames):
        print("Loaded: ", filename)

    database_random = pd.concat(databases)
    mu1_t = database_random['mu1_t'].values
    var1_t = database_random['var1_t'].values
    n1_t = database_random['n1_t'].values
    p1_t = database_random['p1_t'].values
    mu2_t= database_random['mu2_t'].values
    var2_t= database_random['var2_t'].values
    n2_t = database_random['n2_t'].values
    p2_t = database_random['p2_t'].values
    a_t = database_random['a_t'].values
    cdf1_cols = [c for c in database_random if c.startswith('CDF_1_idx_')]
    cdf2_cols = [c for c in database_random if c.startswith('CDF_2_idx_')]
    cdfs_1 = database_random[cdf1_cols].values
    cdfs_2 = database_random[cdf2_cols].values
    print("estrazioni totali:", mu1_t.shape[0])
    return (mu1_t, var1_t,n1_t, p1_t, mu2_t, var2_t,n2_t, p2_t, a_t, cdfs_1, cdfs_2)

def generate_chosen(data, random_vars):
    mu1_t, var1_t,n1_t, p1_t, mu2_t, var2_t,n2_t, p2_t, a_t, cdfs_1, cdfs_2 = random_vars
    max_power = cdfs_1.shape[1]
    S = cdfs_1.shape[0]

    abundance = data[data!=0]
    log2_values = np.ceil(np.log2(abundance)).astype(int)
    data_count = np.bincount(log2_values, minlength=max_power)
    N = len(abundance)

    generated_2 = [generate_hists(cdf, int(round(N*(1-a))), max_power) for a, cdf in zip(a_t, cdfs_2)]
    generated_1 = [generate_hists(cdf, int(round(N*a)), max_power) for a, cdf in zip(a_t, cdfs_1)]
    generated = np.array(generated_1)
    generated += np.array(generated_2)

    skellam = (generated - data_count)**2/(generated + data_count)
    nansum = np.nansum(skellam, axis=1)
    non_nan_count = np.sum(~np.isnan(skellam), axis=1)
    df = non_nan_count-1
    chi_square_sign = st.chi2.cdf(nansum, df)
    chose_skel = (chi_square_sign < 0.5)

    difference = abs(generated - data_count)
    diff_sum = np.sum(difference, axis=1)

    threshold=0.3
    chosen_1 = np.array(diff_sum < np.array([np.ceil(sum(data_count)*threshold)]*len(diff_sum)))
    chosen_2 = np.array(difference<np.array([np.ceil(threshold*data_count)+1]*len(difference)))

    chosen_2=np.array([functools.reduce(mul, chosen_2[l], 1) for l in np.arange(len(chosen_2))])
    chosen = chose_skel*chosen_2.astype(bool)

    result = OrderedDict()
    result['mu1_t'] = mu1_t[chosen]
    result['var1_t'] = var1_t[chosen]
    result['n1_t'] = n1_t[chosen]
    result['p1_t'] = p1_t[chosen]
    result['mu2_t'] = mu2_t[chosen]
    result['var2_t'] = var2_t[chosen]
    result['n2_t'] = n2_t[chosen]
    result['p2_t'] = p2_t[chosen]
    result['a_t'] = a_t[chosen]

    teta2_t = N/((p2_t**(-n2_t)-1.)*gamma(n2_t))
    result['teta2_t'] = teta2_t[chosen]
    teta1_t = N/((p1_t**(-n1_t)-1.)*gamma(n1_t))
    result['teta1_t'] = teta1_t[chosen]
    result['chi_sign'] = chi_square_sign[chosen]
    #result['chi_sign_mio'] = chi_square_sign_mio[chosen]
    result = pd.DataFrame(result)
    return result

# %%
#############################
# ABC Fit of data:
#############################


chosen_database_filename = random_values
otu_table = pd.read_table(filename, index_col="#OTU ID")


print("Database: "+str(chosen_database_filename))
random_vars = read_parameter_distribution(chosen_database_filename)
def abc_fit(sample):
    print("running: {}".format(sample))
    
    data_abundance = otu_table[sample].values
    data_abundance = data_abundance[data_abundance!=0] # Remove zeros
    result = generate_chosen(data_abundance, random_vars)
    # filename = output_folder+"OTU:sample:{}_datetime:{}.csv"
    # result.to_csv(filename.format(sample, datetime.now()), sep='\t')
    filename = output_folder+"OTU_sample_{}.csv"
    result.to_csv(filename.format(sample), sep='\t')
    return sample, len(result)

# abc_fit(otu_table.columns[0]) # if not in parallel

#with Pool(pool_size) as executor:
#    executor.map(abc_fit, otu_table.columns)

for sample_name in otu_table.columns:
    abc_fit(sample_name)

print("DONE")

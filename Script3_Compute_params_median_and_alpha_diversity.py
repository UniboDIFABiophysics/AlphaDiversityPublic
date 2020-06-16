"""
Per ogni campione calcolo mediana e CI dei parametri
"""
import scipy.stats as st
import pymc
import numpy as np
import pandas as pd
import os
import pylab as plt
import numpy.random as rn
import seaborn
import time
from multiprocessing import Pool
import matplotlib as mpl
import glob
import re
mpl.rcParams['axes.unicode_minus']=False

def conf_int(x,lim_inf,lim_sup,len_bins):
    n, bins, patches = plt.hist(x,len_bins, normed=1,histtype='step', cumulative=True)
    if min(n)> 0.5:
        x_mean = bins[0]
    else:
        x_mean =  bins[np.where(n<=0.5)][-1]
    if min(n)< lim_inf:
        x_min = bins[np.where(n<=lim_inf)][-1]
    else:
        x_min = bins[0]
    if min(n)>=lim_sup:
        x_max = bins[-1]
    else:
        x_max = bins[np.where(n<= lim_sup)][-1]
    plt.close()
    return x_mean,x_min,x_max
current_palette = seaborn.color_palette()
current_palette_ = current_palette[1:]
current_palette_.append(current_palette[0])

# %%

############################
# Load otu table:
# Each column corresponds to one subject; each row to one otu.
# The first column contains otu labels and is named "#OTU ID"
# The file is tab separated
############################

filename = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/data/OTUs_UPARSE/pooled_file_Biagisamples.otutab.txt"
otu_table = pd.read_table(filename,index_col="#OTU ID")
data_ = otu_table.T
data_[:3]
#################################
# Read SampleSheet: parameters and alpha diversity will be added to this file
# The first column should contain the sample names (same labels as in the column names of otu_table)
# Should be tab separated
#################################
filename = '/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/data/SampleSheet.txt'
data_legend = pd.read_table(filename)
data_legend = data_legend.rename(columns={data_legend.columns[0]:"SampleID"})
#################################
# Select folder with ABC fit results (output_folder2)
#################################
output_folder2 = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_round2/"
output_folder_results = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_results/"

# %%
#################################
# Read parameters from databases (Fit results)
#################################
os.chdir(output_folder2)
filenames = './OTU*'
filenames = glob.glob(filenames)
my_regex = re.compile('sample:(.*?)_')
samples = [my_regex.findall(name)[0] for name in filenames]
databases = []
for name, ID in zip(filenames, samples):
    db = pd.read_csv(name, sep='\t')
    db['sampleID'] = ID
    databases.append(db)
    #print("caricato: ", name)
database_random = pd.concat(databases)
print('numero dati',len(database_random.index),len(set(database_random.sampleID)))
database_random = database_random.set_index(np.arange(len(database_random.index)))
database_random = database_random[np.isfinite(database_random.teta1_t)]
print('numero dati',len(database_random.index))

### numero iniziale di samples e numero di samples che riesco a fittare:
print(len(otu_table.columns),len(set(database_random.sampleID)))

### Which samples I can't fit:
set(otu_table.columns)-set(database_random.sampleID)

# %%
#################################
# Compute mean and stdev of the paramters:
#################################
variabili_str = database_random.columns[1:-1]
for var_i in variabili_str:
    data_legend[var_i] = 99999.0
    data_legend[var_i+'inf'] = 99999.0
    data_legend[var_i+'sup'] = 99999.0


# os.chdir(output_folder_results)
if not os.path.exists(output_folder_results+"data_legend_per_sampels"):
    os.makedirs(output_folder_results+"data_legend_per_sampels")
def compute_median(sample):
    test = database_random[database_random.sampleID==sample]
    indice = data_legend[data_legend.SampleID==str(sample)].index
    if len(test.index)>1:

        len_bins = 100000
        lim_inf = 0.05
        lim_sup = 0.95

        for colonna in test.columns[1:-1]:
            exec(colonna+' = list(test[colonna])')
            exec(colonna+' = list(np.array('+colonna+')[np.isfinite('+colonna+')])')
            exec(colonna+'_conf = conf_int('+colonna+',lim_inf,lim_sup,len_bins)')
            plt.close()
            exec("data_legend.loc[indice,colonna] = "+colonna+"_conf[0]")
            exec("data_legend.loc[indice,colonna+'inf'] = "+colonna+"_conf[1]")
            exec("data_legend.loc[indice,colonna+'sup'] = "+colonna+"_conf[2]")
    else:
        for colonna in test.columns[1:-1]:
            exec("data_legend.loc[indice,colonna] = test[colonna][test.index[0]]")
            exec("data_legend.loc[indice,colonna+'inf'] = test.loc[test.index[0],colonna]")
            exec("data_legend.loc[indice,colonna+'sup'] = test.loc[test.index[0],colonna]")
    result = data_legend.ix[indice]
    filename = output_folder_results+"data_legend_per_sampels/data_legend_{}.csv"
    result.to_csv(filename.format(sample), sep='\t')
    return result


pool = Pool(10)
pool.map(compute_median, list(set(database_random.sampleID)))
pool.close()
# %%

#############################
# Merge all results in one data_legend file:
#############################
filenames = output_folder_results+"data_legend_per_sampels/data_legend*"
filenames = glob.glob(filenames)

my_regex = re.compile('data_legend_(.*?)_')
samples = [my_regex.findall(name)[0] for name in filenames]

databases = []
for name, ID in zip(filenames, samples):
    db = pd.read_csv(name, sep='\t')
    db['sampleID'] = ID
    databases.append(db)
database_random = pd.concat(databases)
database_random = database_random.drop('Unnamed: 0',1)
filename = output_folder_results+'data_legend_param_median.csv'
database_random.to_csv(filename)

# %%

#############################
# Compute alpha diversity with other indices:
#############################

J = {}
Hill_2 = {}
Hill_1 = {}
Simpson = {}
Shannon = {}
for sample in otu_table.columns:
    data = otu_table[sample].values
    data= data[data!=0]
    S = len(data)
    p = np.array(data,float)
    p = p/np.sum(data)
    H = - np.sum(p*np.log(p))
    H_max = np.log(S)
    Shannon[sample] = H
    Simpson[sample] = np.sum(p**2)#Simpson's index
    J[sample] = H/H_max#Pielou's evenness index
    Hill_1[sample] = np.exp(H)#Hill number 1
    Hill_2[sample] = 1./np.sum(p**2)#hill number 2

alpha_div = pd.DataFrame(index = J.keys(),columns=('Simpson','Shannon','Pielou','Hill_1','Hill_2'))
for i in alpha_div.index:
    alpha_div.loc[i,'Simpson'] = Simpson[i]
    alpha_div.loc[i,'Shannon'] = Shannon[i]
    alpha_div.loc[i,'Pielou'] = J[i]
    alpha_div.loc[i,'Hill_1'] = Hill_1[i]
    alpha_div.loc[i,'Hill_2'] = Hill_2[i]

filename = output_folder_results+ 'data_legend_param_median.csv'
data_legend = pd.read_csv(filename)
data_legend = data_legend.set_index('SampleID')

for i in data_legend.index:
    for colonna in alpha_div.columns:
        data_legend.loc[i,colonna] = alpha_div.loc[str(i),colonna]
data_legend.columns[0]
data_legend = data_legend.drop('Unnamed: 0', axis=1)
filename = output_folder_results+'data_legend_param_median_alpha.csv'
data_legend.to_csv(filename)

# %%

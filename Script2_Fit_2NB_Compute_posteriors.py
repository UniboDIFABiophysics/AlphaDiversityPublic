#!/home/PERSONALE/claudia.sala3/SHARED_FOLDER/miniconda3/bin python

import scipy.stats as st
import numpy as np
import pandas as pd
import pylab as plt
from scipy.special import gamma
from collections import OrderedDict
from datetime import datetime
import matplotlib as mpl
import os
import seaborn
import pymc
import glob
import re


############################
# Load otu table:
# Each column corresponds to one subject; each row to one otu.
# The first column contains otu labels and is named "#OTU ID"
# The file is tab separated
############################

filename = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/data/OTUs_UPARSE/pooled_file_Biagisamples.otutab.txt"
otu_table = pd.read_table(filename,index_col="#OTU ID")

############################
# Choose round and folders:
############################
round_i = 1

output_folder1 = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_round1/"
output_folder2 = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_round2/"
priors_folder1 = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_priors_round1/" # folder with priors for round 1 (and parameter estimates)
priors_folder2 = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_priors_round2/" # folder with priors for round 2

if round_i ==1:
    exp_i = 2
    priors_folder = priors_folder1
    output_folder = output_folder1

if round_i ==2:
    exp_i = 1
    priors_folder = priors_folder2
    output_folder = output_folder2
# %%
#############################
# PRIOR PROBABILITY
#############################

if round_i == 1:
    mu1_dist = st.invgamma(1.5,1.).rvs
    var1_dist = st.invgamma(1.5,1.).rvs
    mu2_dist = st.invgamma(1.5,1.).rvs
    var2_dist = st.invgamma(1.5,1.).rvs
    a_dist = st.uniform(0.,1.).rvs
if round_i == 2:
    filename = priors_folder1+'parametri.csv' # parameters computed from the round 1 posteriors are round 2 priors
    parametri = pd.read_csv(filename,index_col='Unnamed: 0')
    mu1_dist =st.gamma(parametri.loc['a','mu1_t'] ,scale= parametri.loc['b','mu1_t'] ).rvs
    var1_dist =st.gamma( parametri.loc['a','var1_t'] ,scale= parametri.loc['b','var1_t']  ).rvs
    mu2_dist =st.gamma( parametri.loc['a','mu2_t'] ,scale= parametri.loc['b','mu2_t'] ).rvs
    var2_dist =st.gamma( parametri.loc['a','var2_t'] ,scale= parametri.loc['b','var2_t']  ).rvs
    a_dist=st.beta( parametri.loc['a','a_t'], parametri.loc['b','a_t'] ).rvs

# Simulate data from priors:
a_ipo = a_dist(500)
p1=[]
p2=[]
mu1=[]
mu2=[]
n1=[]
n2=[]
var1=[]
var2=[]
titoli = [r'$\mu_1$',r'$\sigma^2_1$',r'$\mu_2$',r'$\sigma^2_2$']
for i in np.arange(500):
    mu1_t = mu1_dist()
    var1_t = mu1_t+var1_dist()**exp_i ### lo scrivo così perchè per avere p <1 deve essere mu < var; in round_1 ho **2
    var1.append(var1_t)
    mu1.append(mu1_t)
    p1.append(mu1_t/var1_t)
    #p1.append(n1_t/(n1_t+mu1_t))
    n1.append((mu1_t**2)/(var1_t-mu1_t))

    mu2_t = mu1_t+mu2_dist()
    var2_t = mu2_t+var2_dist()**exp_i
    var2.append(var2_t)
    mu2.append(mu2_t)
    p2.append(mu2_t/var2_t)
    #n2.append(mu2_t**2/(var2_t-mu2_t))

variabili = mu1,(var1),mu2,(var2)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ii=0
for fig_i in np.arange(2):
    for fig_j in np.arange(2):
        x_val = variabili[ii]
        seaborn.distplot((x_val), bins=20, ax=ax[fig_i][fig_j])
        ax[fig_i][fig_j].set_title(titoli[ii],fontsize=15)
        ii+=1

plt.show()


# %%
#############################
# Read results to compute posterior distribution
#############################

lista = otu_table.columns
os.chdir(output_folder)

filenames = './OTU*'
filenames = glob.glob(filenames)

my_regex = re.compile('sample:(.*?)_')
samples = [my_regex.findall(name)[0] for name in filenames]
len(samples)*10
databases = []
for name, ID in zip(filenames, samples):
    db = pd.read_csv(name, sep='\t')
    db['sampleID'] = ID
    ### per fare i fit dei parametri prendo solo i primi parametri tenuti per ogni campioni (5*177=885):
    databases.append(db[:10]) # TODO: selezionare il numero di dati otttenuti per campione
    ### per guardare la probabilità del modello invece li tengo tutti
    #databases.append(db)


database_random = pd.concat(databases)
print( 'numero dati',len(database_random.index))
print( 'number of found samples',len(set(database_random.sampleID)))
print( len(list(set(lista)-set(database_random.sampleID))),"samples not found")
# set(lista)-set(database_random.sampleID)

# %%
parametri = pd.DataFrame(index = ('a','b'),columns = ('mu1_t','var1_t','mu2_t','var2_t','a_t'))

pymc_distributions = {'a_t': pymc.Beta,
                      'mu1_t': pymc.Gamma,
                      'mu2_t': pymc.Gamma,
                      'var1_t': pymc.Gamma,
                      'var2_t': pymc.Gamma,
                      }
scipy_distributions = {'a_t': lambda a, b: st.beta.rvs(a, b),
                       'mu1_t': lambda a, b: st.gamma.rvs(a, scale=1/b),
                       'mu2_t': lambda a, b: st.gamma.rvs(a, scale=1/b),
                       'var1_t': lambda a, b: st.gamma.rvs(a, scale=1/b),
                       'var2_t': lambda a, b: st.gamma.rvs(a, scale=1/b),
                       }
# %%
variabili_ = ['mu1_t','var1_t','mu2_t','var2_t']
param_ = [mu1,var1,mu2,var2]
for var_i in np.arange(2):
    variable = variabili_[var_i]
    param = param_[var_i]
    groups = {k:group for k, group in database_random.groupby('sampleID')[variable]}
    a = pymc.Uninformative('a', value=1)
    b = pymc.Uninformative('b', value=1)
    variables = [a, b]
    distribution = pymc_distributions[variable]
    for k, g in groups.items():
        obs = distribution('obs{}'.format(k),
                           alpha = a,
                           beta = b,
                           observed=True,
                           value=g.values)
        variables.append(obs)

    model_map = pymc.MAP(variables)
    model_map.fit()
    model_mcmc = pymc.MCMC(variables)
    model_mcmc.sample(1e5)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    seaborn.distplot(a.trace(), ax=ax1)
    #ax1.axvline(model_mcmc.stats()['a']['mean'])
    ax1.axvline(a.trace().mean())
    seaborn.distplot(b.trace(), ax=ax2)
    #ax2.axvline(model_mcmc.stats()['b']['mean'])
    ax2.axvline(b.trace().mean())
    seaborn.boxplot(y=variable, x='sampleID', data=database_random, ax=ax4, notch=True)
    plt.setp(ax4.get_xticklabels(), rotation='vertical')
    histo = ax3.hist(param,bins=np.linspace(0.,max(database_random[variable]),100),alpha=0.5,color='g')
    #ax3.hist(st.gamma.rvs(model_mcmc.stats()['a']['mean'], scale=1./model_mcmc.stats()['b']['mean'],
    #                      size=np.sum(histo[0])),bins=np.linspace(0.,max(database_random[variable]),100),alpha=0.5,color='b')
    ax3.hist(st.gamma.rvs(a.trace().mean(), scale=1./b.trace().mean(),
                          size=int(np.sum(histo[0]))),bins=np.linspace(0.,max(database_random[variable]),100),alpha=0.5,color='b')

    #print( '\n',variable,'=st.gamma(',model_mcmc.stats()['a']['mean'],',scale=', 1./model_mcmc.stats()['b']['mean'],').rvs')
    #parametri.loc['a',variable] = model_mcmc.stats()['a']['mean']
    #parametri.loc['b',variable] = 1./model_mcmc.stats()['b']['mean']
    print( '\n',variable,'=st.gamma(',a.trace().mean(),',scale=', 1./b.trace().mean(),').rvs')
    parametri.loc['a',variable] = a.trace().mean()
    parametri.loc['b',variable] = 1./b.trace().mean()
    plt.show()
    fig.savefig(priors_folder+"posteriors_round"+str(round_i)+"_"+variable+".png",dpi=200,bbox_inches="tight")

for var_i in range(2,4):
    variable = variabili_[var_i]
    param = param_[var_i]
    groups = {k:group for k, group in database_random.groupby('sampleID')[variable]}
    a = pymc.Uninformative('a', value=1)
    b = pymc.Uninformative('b', value=1)
    variables = [a, b]
    distribution = pymc_distributions[variable]
    for k, g in groups.items():
        obs = distribution('obs{}'.format(k),
                           alpha = a,
                           beta = b,
                           observed=True,
                           value=g.values)
        variables.append(obs)

    model_map = pymc.MAP(variables)
    model_map.fit()
    model_mcmc = pymc.MCMC(variables)
    model_mcmc.sample(1e5)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    seaborn.distplot(a.trace(), ax=ax1)
    #ax1.axvline(model_mcmc.stats()['a']['mean'])
    ax1.axvline(a.trace().mean())
    seaborn.distplot(b.trace(), ax=ax2)
    #ax2.axvline(model_mcmc.stats()['b']['mean'])
    ax2.axvline(b.trace().mean())
    seaborn.boxplot(y=variable, x='sampleID', data=database_random, ax=ax4, notch=True)
    plt.setp(ax4.get_xticklabels(), rotation='vertical')
    histo = ax3.hist(param,bins=np.linspace(0.,max(database_random[variable]),100),alpha=0.5,color='g')
    #ax3.hist(st.gamma.rvs(model_mcmc.stats()['a']['mean'], scale=1./model_mcmc.stats()['b']['mean'],
    #                      size=np.sum(histo[0])),bins=np.linspace(0.,max(database_random[variable]),100),alpha=0.5,color='b')
    ax3.hist(st.gamma.rvs(a.trace().mean(), scale=1./b.trace().mean(),
                          size=int(np.sum(histo[0]))),bins=np.linspace(0.,max(database_random[variable]),100),alpha=0.5,color='b')

    #print( '\n',variable,'=st.gamma(',model_mcmc.stats()['a']['mean'],',scale=', 1./model_mcmc.stats()['b']['mean'],').rvs')
    #parametri.loc['a',variable] = model_mcmc.stats()['a']['mean']
    #parametri.loc['b',variable] = 1./model_mcmc.stats()['b']['mean']
    print( '\n',variable,'=st.gamma(',a.trace().mean(),',scale=', 1./b.trace().mean(),').rvs')
    parametri.loc['a',variable] = a.trace().mean()
    parametri.loc['b',variable] = 1./b.trace().mean()
    plt.show()
    fig.savefig(priors_folder+"posteriors_round"+str(round_i)+"_"+variable+".png",dpi=200,bbox_inches="tight")

variable = 'a_t'
param = a_ipo
groups = {k:(group) for k, group in database_random.groupby('sampleID')[variable]}
a = pymc.Uninformative('a', value=1)
b = pymc.Uninformative('b', value=1)
variables = [a, b]
distribution = pymc_distributions[variable]
for k, g in groups.items():
    obs = distribution('obs{}'.format(k),
                       alpha = a,
                       beta = b,
                       observed=True,
                       value=g.values)
    variables.append(obs)

model_mcmc = pymc.MCMC(variables)
model_mcmc.sample(1e5)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
seaborn.distplot(a.trace(), ax=ax1)
ax1.axvline(a.trace().mean())
seaborn.distplot(b.trace(), ax=ax2)
ax2.axvline(b.trace().mean())
seaborn.boxplot(y=variable, x='sampleID', data=database_random, ax=ax4, notch=True)
plt.setp(ax4.get_xticklabels(), rotation='vertical')

histo = ax3.hist(param,bins=np.linspace(0.,1.,15.),alpha=0.5,color='g')
ax3.hist(st.beta.rvs(a.trace().mean(),b.trace().mean(),
                      size=int(sum(histo[0]))),bins=np.linspace(0.,1.,15.),alpha=0.5,color='b')
print( '\n','a_dist=st.beta(',a.trace().mean(),',', b.trace().mean(),').rvs')
parametri.loc['a',variable] = a.trace().mean()
parametri.loc['b',variable] = b.trace().mean()
plt.show()
fig.savefig(priors_folder+"posteriors_round"+str(round_i)+"_"+variable+".png",dpi=200,bbox_inches="tight")
# %%
# #  Save parameters:
filename = priors_folder+'parametri.csv'
parametri.to_csv(filename)

parametri

# %%

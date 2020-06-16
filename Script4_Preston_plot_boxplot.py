import scipy.stats as st
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
def bins_log2(data):
    num_bins = int(round(np.log2(max(data)))+1) ###il numero di bins è dato dai dati osservati
    return np.logspace(0.0,num_bins ,num=num_bins+1,base=2.0)

def fmedio_nb1(i,a,s,m):
    return N*a*(st.nbinom.cdf(bin_max[i]-1.,s,m)-st.nbinom.cdf(bin_min[i]-1.,s,m))/(1.-m**s)

def fmedio_nb(i,a,s,m,s1,m1):
    return N*(a*(st.nbinom.cdf(bin_max[i]-1.,s,m)-st.nbinom.cdf(bin_min[i]-1.,s,m))/(1.-m**s)+
              (1.-a)*(st.nbinom.cdf(bin_max[i]-1.,s1,m1)-st.nbinom.cdf(bin_min[i]-1.,s1,m1))/(1.-m1**s1))

def conf_int(x,lim_inf,lim_sup,len_bins):
    n, bins, patches = plt.hist(x,len_bins, normed=1,histtype='step', cumulative=True)
    if min(n)> 0.5:
        x_mean = bins[0]
    else:
        x_mean =  bins[n<=0.5][-1]
    if min(n)< lim_inf:
        x_min = bins[(n<= lim_inf)][-1]
    else:
        x_min = bins[0]
    if min(n)>=lim_sup:
        x_max = bins[-1]
    else:
        x_max = bins[(n<= lim_sup)][-1]
    plt.close()
    return x_mean,x_min,x_max

current_palette = seaborn.color_palette()
current_palette_ = current_palette[1:]
current_palette_.append(current_palette[0])

# %%
##################################
# Read otu table and ABC fit results
##################################
filename = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/data/OTUs_UPARSE/pooled_file_Biagisamples.otutab.txt"
otu_table = pd.read_table(filename,index_col="#OTU ID")
data_ = otu_table.T  # samples sulle righe, otu sulle colonne
data_[:1]

output_folder_results = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_results/"
filename = output_folder_results+'data_legend_param_median_alpha.csv'
data_legend = pd.read_csv(filename)
data_legend = data_legend.set_index("SampleID") # I created this column named SampleID in Script3

output_folder2 = "/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Biagi_centenari/Risultati_ABC_prova/ABC_round2/"

# %%
# Average number of reads per sample:
np.mean(otu_table.sum(axis=0))
np.max(otu_table.sum(axis=0))
np.std(otu_table.sum(axis=0))
len(otu_table.index)
# %%
# Read parameters from databases (Fit results)
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
set(database_random.sampleID)
# %%
###################################
# show preston plot:
# NB1 = blue
# NB2 = magenta
###################################

samples = list(set(database_random.sampleID).intersection(set(data_legend[data_legend["n1_t"]<0.01].index)))
len(samples)
# %%
# Plot di esempio per il paper (da Kong)
sample = samples[0] # "SRR3679961"
test = database_random[database_random.sampleID==sample][:1000]
data = otu_table[sample]
data = data[data!=0]
len_bins = len(test.index)

bins = bins_log2(data)
a_t = list(test.a_t)
n2_t = list(test.n2_t)
mu2_t = list(test.mu2_t)
p2_t = list(test.p2_t)
teta2_t = list(test.teta2_t)
n1_t = list(test.n1_t)
mu1_t = list(test.mu1_t)
p1_t = list(test.p1_t)
teta1_t = list(test.teta1_t)

fig,ax=plt.subplots()
hist_ = plt.hist(data, bins=bins_log2(data),color='gray',alpha=0.6)
yn_ = hist_[0]
xn_ = np.empty(len(yn_))
bin_min = np.empty(len(yn_))
bin_max = np.empty(len(yn_))
index_x = np.arange(len(yn_))
N = sum(hist_[0])
for i in range (0,len(hist_[1])-1):
    xn_[i]=hist_[1][i]+((hist_[1][i+1]-hist_[1][i])/2)
    bin_min[i] = hist_[1][i]
    bin_max[i]= hist_[1][i+1]
plt.semilogx(basex=2)
risultati_nb = pd.DataFrame(columns=np.arange(len(yn_)))
for j in np.arange(len(a_t)):
    if (np.isfinite(a_t[j]))&(np.isfinite(n2_t[j]))&(np.isfinite(p2_t[j]))&(np.isfinite(n1_t[j]))&(np.isfinite(p1_t[j])):
        risultati_nb.loc[j] = list(fmedio_nb(np.arange(len(yn_)), a_t[j],n1_t[j],p1_t[j],n2_t[j],p2_t[j]))
bp = plt.boxplot([list(risultati_nb[i]) for i in risultati_nb.columns],positions=xn_,widths = (bin_max-bin_min)*50./100.)

plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='black', marker='+')
plt.xlim(0.0,bin_max[-1])
risultati_nb = fmedio_nb(np.arange(len(yn_)), np.median(np.array(a_t)), np.median(np.array(n1_t)),
                           np.median(np.array(p1_t)), np.median(np.array(n2_t)), np.median(np.array(p2_t)))
risultati_nb1 =fmedio_nb1(np.arange(len(yn_)), np.median(np.array(a_t)),
                           np.median(np.array(n1_t)), np.median(np.array(p1_t)))
risultati_nb2 = fmedio_nb1(np.arange(len(yn_)), 1.-np.median(np.array(a_t)), np.median(np.array(n2_t)),
                           np.median(np.array(p2_t)))
plt.plot(xn_,risultati_nb1,'b-', linewidth=1.1, label="Rare")
plt.plot(xn_,risultati_nb2,'m-', linewidth=1.1,label="Abundant")
plt.xlabel('Abundance Category',fontsize=20)
plt.ylabel('Number of Species',fontsize=20)
plt.semilogx(basex=2)
# plt.title(str(sample),fontsize=20)
plt.setp(ax.get_xticklabels(),fontsize=16)
plt.setp(ax.get_yticklabels(),fontsize=16)
# plt.legend(loc="best")
plt.show()
# filename = '/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Kong/Risultati_ABC/preston_'+sample+'_'+str(len(np.array(a_t)))+'params_2nb_example.png'
# fig.savefig(filename,dpi=300,bbox_inches='tight')
# filename = '/mnt/stor-rw/users/claudia.sala3/GM_diversity/dati_Kong/Risultati_ABC/preston_'+sample+'_'+str(len(np.array(a_t)))+'params_2nb_example.svg'
# fig.savefig(filename,dpi=300,bbox_inches='tight')


# %%
p_values_chi_square = []
# sample = "SRR3679961"
for sample in samples:
    test = database_random[database_random.sampleID==sample][:1000]
    data = otu_table[sample]
    data = data[data!=0]
    len_bins = len(test.index)

    bins = bins_log2(data)
    a_t = list(test.a_t)
    n2_t = list(test.n2_t)
    mu2_t = list(test.mu2_t)
    p2_t = list(test.p2_t)
    teta2_t = list(test.teta2_t)
    n1_t = list(test.n1_t)
    mu1_t = list(test.mu1_t)
    p1_t = list(test.p1_t)
    teta1_t = list(test.teta1_t)

    fig,ax=plt.subplots()
    hist_ = plt.hist(data, bins=bins_log2(data),color='b',alpha=0.3)
    yn_ = hist_[0]
    xn_ = np.empty(len(yn_))
    bin_min = np.empty(len(yn_))
    bin_max = np.empty(len(yn_))
    index_x = np.arange(len(yn_))
    N = sum(hist_[0])
    for i in range (0,len(hist_[1])-1):
        xn_[i]=hist_[1][i]+((hist_[1][i+1]-hist_[1][i])/2)
        bin_min[i] = hist_[1][i]
        bin_max[i]= hist_[1][i+1]
    plt.semilogx(basex=2)
    risultati_nb = pd.DataFrame(columns=np.arange(len(yn_)))
    for j in np.arange(len(a_t)):
        if (np.isfinite(a_t[j]))&(np.isfinite(n2_t[j]))&(np.isfinite(p2_t[j]))&(np.isfinite(n1_t[j]))&(np.isfinite(p1_t[j])):
            risultati_nb.loc[j] = list(fmedio_nb(np.arange(len(yn_)), a_t[j],n1_t[j],p1_t[j],n2_t[j],p2_t[j]))
    bp = plt.boxplot([list(risultati_nb[i]) for i in risultati_nb.columns],positions=xn_,widths = (bin_max-bin_min)*50./100.)

    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.xlim(0.0,bin_max[-1])
    risultati_nb = fmedio_nb(np.arange(len(yn_)), np.median(np.array(a_t)), np.median(np.array(n1_t)),
                               np.median(np.array(p1_t)), np.median(np.array(n2_t)), np.median(np.array(p2_t)))
    risultati_nb1 =fmedio_nb1(np.arange(len(yn_)), np.median(np.array(a_t)),
                               np.median(np.array(n1_t)), np.median(np.array(p1_t)))
    risultati_nb2 = fmedio_nb1(np.arange(len(yn_)), 1.-np.median(np.array(a_t)), np.median(np.array(n2_t)),
                               np.median(np.array(p2_t)))

    plt.plot(xn_,risultati_nb,'k--', linewidth=1,label="Total")
    plt.plot(xn_,risultati_nb1,'b-', linewidth=1, label="Rare")
    plt.plot(xn_,risultati_nb2,'m-', linewidth=1,label="Abundant")
    plt.xlabel('Abundance Category',fontsize=15)
    plt.ylabel('Number of Species',fontsize=15)
    plt.semilogx(basex=2)
    plt.title(str(sample),fontsize=12)
    plt.legend(loc="best")
    # plt.close()
    plt.show()
    # print( 'time:',(time.time()-start)/60.)
    #filename = '/home/PERSONALE/claudia.sala3/' + studio + '/Risultati_ABC/results_figure/preston_plots_boxplot/preston_'+sample+'_'+str(len(np.array(a_t)))+'params_2nb.png'
    #fig.savefig(filename,dpi=200,bbox_inches='tight')
    p_values_chi_square.append(st.chisquare(yn_, risultati_nb)[1])

# %%
# Histogram of p_values_chi_square

p_values_chi_square = np.array(p_values_chi_square,float)
len(p_values_chi_square)
np.sum(p_values_chi_square[p_values_chi_square>0]<0.05)
plt.hist(p_values_chi_square, bins=np.linspace(0, 1, 21))
plt.show()
# %%
# Plot with only regression lines:
risultati_nb = fmedio_nb(np.arange(len(yn_)), np.median(np.array(a_t)), np.median(np.array(n1_t)),
                           np.median(np.array(p1_t)), np.median(np.array(n2_t)), np.median(np.array(p2_t)))
risultati_nb1 =fmedio_nb1(np.arange(len(yn_)), np.median(np.array(a_t)),
                           0.4, 0.1)
risultati_nb2 = fmedio_nb1(np.arange(len(yn_)), np.median(np.array(a_t)),
                           0.4, 0.01)

plt.plot(xn_,risultati_nb,'k--', linewidth=1,label="Total")
plt.plot(xn_,risultati_nb1,'b-', linewidth=1, label="Rare")
plt.plot(xn_,risultati_nb2,'m-', linewidth=1,label="Abundant")
plt.semilogx(basex=2)
plt.title(str(sample),fontsize=12)
plt.legend(loc="best")
plt.show()

# AlphaDiversity2NB

SampleSheet.txt and pooled_file_Biagisamples.otutab.txt are two example files containing the samples metadata and the otu table (otu abundances).

Script1_Fit_2NB_model_round1.py is used to sample the 2NB model parameters from their prior distributions. The priors hyperparameters are defined so that to have broad distributions (round 1 of the ABC fit) or based on the parameters posterior distributions (round2 of the ABC fit).
Moreover, Script1 is used to performe the ABC fit.

Script2_Fit_2NB_Compute_posteriors.py is used to obtain the paramters posterior distributions starting from the results of the ABC fit.

Script3_Compute_params_median_and_alpha_diversity.py is used to compute the median of the model parameters obtained in the ABC fit. Moreover, in Script3 we compute the classical alpha diversity indices (e.g. Simpson, Shannon, etc.) based on the otu relative abundances.

Script4_Preston_plot_boxplot.py	can be used to plot for each sample the Preston's plot and the fitted curve.

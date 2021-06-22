# ckd_markov_model.py
# JMA 22 June 2021
# Simulate the chronic kidney disease markov ckd_markov_model

import os, sys
import numpy as np
import pandas as pd

# Note, as read this is column markov - the columns sum to 100.  
ckd_m = pd.read_csv('ckd_transition_matrix.tab', sep='\t', header=2, index_col=0)
ckd_rowmarkov = np.transpose(ckd_m.values)
# Normalize to P.s
row_sums = np.apply_along_axis(lambda x: 1/sum(x), axis=0, arr=ckd_rowmarkov)

# The result is a matrix with rows that sum to one. 
ckd_normalized = np.multiply(ckd_rowmarkov, row_sums)

def forward_simulation(init_state =np.array([1,0,0,0,0,0]), total_months=12):
	pass

print( ckd_m.index)
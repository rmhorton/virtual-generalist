# ckd_markov_model.py
# JMA 22 June 2021
# Simulate the chronic kidney disease markov ckd_markov_model

import os, sys
import numpy as np
import pandas as pd


from bokeh.plotting import figure, show, save
# from bokeh.layouts import column, row, gridplot
from bokeh.models import Band, BoxAnnotation, ColumnDataSource, FixedTicker, Label
from bokeh import palettes

# Note, as read this is column markov - the columns sum to 100.  
ckd_m = pd.read_csv('./BN/ckd_transition_matrix.tab', sep='\t', header=2, index_col=0)
ckd_cols = ckd_m.columns
print(ckd_cols)
ckd_rowmarkov = np.transpose(ckd_m.values)
# Normalize to P.s
row_sums = np.apply_along_axis(lambda x: 1/sum(x), axis=0, arr=ckd_rowmarkov)

# The result is a matrix with rows that sum to one. 
ckd_normalized = np.multiply(ckd_rowmarkov, row_sums)

def forward_simulation(init_state =np.array([1,0,0,0,0,0]), total_months=180):
	trajectory = init_state.reshape((1,len(init_state)))
	for k in range(total_months):
		current_month = np.matmul(ckd_normalized, trajectory[k,:])
		trajectory = np.vstack([trajectory, current_month])
	return trajectory

def sim_plt(traj_df):
	one_fig = figure(plot_width = 1400, plot_height = 400, 
		y_range=[-0.02,1.02], 
		y_axis_label = 'Probability',
		x_axis_label = 'Years',
		title=f'Progression of the ckd states over years')
	for k, a_col in enumerate(traj_df.columns):
		computed_lc = palettes.Category10_10[k-1]
		a_ts = pd.DataFrame(traj_df[a_col])
		#a_ts.reset_index(inplace=True)
		a_ts['year']  = a_ts.index/12
		cds = ColumnDataSource(a_ts)
		one_fig.line(x='year', y=a_col, source=cds, legend_label=a_col, line_color=computed_lc)
	return one_fig


sim = forward_simulation()
save(sim_plt(pd.DataFrame(sim, columns=ckd_cols)), 'ckd_progression.html', title='ckd_progression')

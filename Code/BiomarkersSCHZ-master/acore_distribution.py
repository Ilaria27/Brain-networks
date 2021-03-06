import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import itertools
from paths import * 
import argparse
from functions import *
import json

"""
Parameters:

connectivity: Multimodal
resolution:   83 , 129,  234
type:         Structural, Fuctional (from the Multimodal)

example,

python acore_distribution.py -connectivity Multimodal -resolution 83 -type Structural

"""
mpl.interactive(True)
np.set_printoptions(linewidth=999999)
plt.close('all')
plt.style.use('ggplot')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-connectivity', dest='connectivity', help='structural or functional')
    parser.add_argument('-resolution', dest='resolution', help='dimension')		
    parser.add_argument('-type', dest='type', help='type')		
    return parser.parse_args()	

########################################################################################################
args = parse_args()

connectivity = args.connectivity #  Multimodal
resolution   = args.resolution   #  83 , 129,  234
type_mode    = args.type         # Structural, Fuctional (from the Multimodal)

# Matrix of selected features given the best stability and accuracy scores
if type_mode == 'Structural':
    rank_mat     = 'A_sc19'     
elif type_mode == 'Functional':   
    rank_mat     = 'A_fc19'
    
num_examples     = 54
data_mat 	     = scipy.io.loadmat(SAVING_MAT+'GlobalResults_'+resolution+'_'+connectivity+'_accuracy.mat')
df_selected_subj = pd.read_csv(SELECTED_SUBJECTS27, index_col=0)


if   (resolution=='83' or resolution=='68'): idx=0; 
elif (resolution=='129' or resolution=='114'): idx=1;
elif (resolution=='234' or resolution=='219'): idx=2;

if connectivity=="Structural":
	prefix = 'sc_'+resolution

elif connectivity=="Functional":
	prefix = 'fc_'+resolution

elif connectivity=="Multimodal":
	multi_prefix = rank_mat[2:4]
	prefix = 'mm_'+resolution
	
#----------------------------------------------------------------------------	
rank_matrices 	=  data_mat[prefix+'_'+'Matrices']
R_rank   		=  rank_matrices[0][rank_mat][0]	
R_rank2   		=  rank_matrices[0]['A_fc'+rank_mat[4:]][0]	

mask            =  (R_rank>50)*1

mat_metadata	 = scipy.io.loadmat(ID_NODES_PATH) 				# Load id names
df_node_labels   = get_node_labels(mat_metadata,idx) # getting node labels based in id

core 			 =  set(list(zip(*nx.Graph(R_rank).edges())[0]))
#----------------------------------------------------------------------------
print(resolution);print(rank_mat);print(prefix);



R_rank_filtered        = np.multiply(mask,R_rank)
df_node_labels['mask'] = R_rank_filtered.sum(axis=1) 
df_grouped             = df_node_labels.sort_values('mask').groupby('hemisphere')

plt.figure()
plt.suptitle("Node strenghts")

a_core = []
for i, (grouped_name, grouped_gdf) in enumerate(df_grouped):
	
	# Horizontal
	ax = plt.subplot(2, 1, i + 1) # nrows, ncols, axes position
	barlist = grouped_gdf.sort_values(by='node_label', ascending=True).plot(kind='bar', fontsize=10, grid=False, cmap='Set2', ax=ax, x='node_label') # Paired, Set2
	plt.hlines(grouped_gdf['mask'].mean(), xmin=-1, xmax=80, linestyles='dashed')

	ax.set_facecolor((1.0, 1.0, 1.0))
	ax.set_title(grouped_name)

	a_core = a_core + list(grouped_gdf[grouped_gdf['mask']>=grouped_gdf['mask'].mean()].index.values)
 
plt.tight_layout()
plt.show()

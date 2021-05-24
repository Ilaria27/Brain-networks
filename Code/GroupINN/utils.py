import numpy as np
import pandas as pd
from sklearn import tree
#from sklearn.tree.export import export_text
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix



def open_file(file, limit, binary, absolute):
    mat = np.array(pd.read_csv(file, header = None, sep = ","))
    
    if absolute:
        mat = abs(mat)
    
    if limit:
        threshold = np.percentile(mat[np.tril_indices_from(mat, k=-1)], limit)
        mat[mat<threshold] = 0
 
    if binary:
        mat[mat>=threshold] = 1

    return mat
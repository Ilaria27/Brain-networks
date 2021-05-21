# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:32:55 2021

@author: Ilaria T
"""

import numpy as np
import pandas as pd
import csv
import os

files = os.listdir("C:\\Users\\Ilaria T\\Desktop\\Sapienza\\Tesi Brain Network\\Code\\Dataset\\children\\asd")
files[0]
matrices = []
for i in files:
    matrices.append(np.loadtxt(open("C:\\Users\\Ilaria T\\Desktop\\Sapienza\\Tesi Brain Network\\Code\\Dataset\\children\\asd\\" + i, "rb"), 
           delimiter=","))
matrices[0]
vec_mat = []
for i in matrices:
    A = matrices[0]
    indices = np.triu_indices_from(i)
    vec_mat.append(np.asarray(i[indices]))
len(vec_mat[0])



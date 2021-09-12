# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:53:41 2021

@author: Ilaria T
"""

import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', help = 'Dataset', type = str)
parser.add_argument('-c1', help = 'Class 1 of dataset', default = 'asd', type = str)
parser.add_argument('-c2', help = 'Class 2 of dataset', default = 'td', type = str)
args = parser.parse_args()

#import file names of patients matrices
#asd
dir1 = "Dataset/{}/{}/".format(args.d, args.c1)
#dir1 = "Dataset/children/asd/"
#dir1 = "Dataset/ABIDE/asd/"
#dir1 = "Dataset/schizo/schz/"
#dir1 = "Dataset/mta/asd/"

c1 = ["{}{}".format(dir1,elem) for elem in os.listdir(dir1)]

#td
dir2 = "Dataset/{}/{}/".format(args.d, args.c2)

#dir2 = "Dataset/children/td/"
#dir2 = "Dataset/ABIDE/td/"
#dir2 = "Dataset/schizo/ctrl/"
#dir2 = "Dataset/mta/td/"

c2 = ["{}{}".format(dir2,elem) for elem in os.listdir(dir2)]

#import all adjacency matrices and transform the edge between a node and itself
#from 1 (if it is 1) to 0, to not calculate that edge.

matrices = dict()
for file in c1:
    f = pd.read_csv(file, index_col = False, header = None)
    f = np.asmatrix(f)
    if f[0,0] == 1:
        for j in range(len(f)):
            f[j,j] = 0
    q1 = np.quantile(f, 0.2)
    q2 = np.quantile(f, 0.8)
    for u in range(len(f)):
        for v in range(len(f.transpose())):
            if f[u,v] < q1:
                f[u,v] = 0
            elif f[u,v] > q2:
                f[u,v] = 0
    matrices[file] = f

    #f = np.asmatrix(f)
    
    #matrices[file] = f
for file in c2:
    f = pd.read_csv(file, index_col = False, header = None)
    f = np.asmatrix(f)
    if f[0,0] == 1:
        for j in range(len(f)):
            f[j,j] = 0
    q1 = np.quantile(f, 0.2)
    q2 = np.quantile(f, 0.8)
    for u in range(len(f)):
        for v in range(len(f.transpose())):
            if f[u,v] < q1:
                f[u,v] = 0
            elif f[u,v] > q2:
                f[u,v] = 0
    matrices[file] = f

#1 autism 2 control
if args.c1 == 'asd':
    classes = []
    for i in range(len(c1)):
        classes.append(1)
    for i in range(len(c2)):
        classes.append(2)
elif args.c1 == 'td':
    classes = []
    for i in range(len(c1)):
        classes.append(2)
    for i in range(len(c2)):
        classes.append(1)
    
#calculate the quantiles 0.2 and 0.8 of each graph (matrix)
'''
q1 = []
q2 = []
for i in matrices.keys():
    q1.append(np.quantile(matrices[i], 0.2))
    q2.append(np.quantile(matrices[i], 0.8))
#take the mean of the 2 quantiles
mq1 = round(np.mean(q1), 3)
mq2 = round(np.mean(q2), 3)

#leave out the values lower of quantile 0.2 and bigger of quantile 0.8
for f in matrices.keys():
    m = matrices[f]
    for i in range(len(m)):
        for j in range(len(m.transpose())):
            if m[i,j] < mq1:
                m[i,j] = 0
            elif m[i,j] > mq2:
                m[i,j] = 0
    matrices[f] = m

for i in matrices.keys():
    m = matrices[i]
    q1 = np.quantile(m, 0.2)
    q2 = np.quantile(m, 0.8)
    for u in range(len(m)):
        for v in range(len(m.transpose())):
            if m[u,v] < q1:
                m[u,v] = 0
            elif m[u,v] > q2:
                m[u,v] = 0
    matrices[i] = m
'''


'''
import xlrd
filepath = 'contrast-subgraph/features.xls'
excel_workbook = xlrd.open_workbook(filepath)
excel_sheet = excel_workbook.sheet_by_index(0)

relevantData = []
for row in range(excel_sheet.nrows):
    if row % 2 != 0:
        rowToArray = list(map(str,excel_sheet.cell_value(row,0).split('[|n|\|]')))
        relevantData.append(rowToArray)
   
relevantData2 = relevantData
for rows in range(len(relevantData)):
    relevantData2[rows].replace('\n', ' ')
    
fottiti = relevantData2[0][0]
fottiti2volte = fottiti.replace('\n', ' ')
fottiti3volte = fottiti2volte.replace('  ', ' ')
fottiti3volte = fottiti3volte.replace('  ', ' ')
fottiti4volte = fottiti3volte.replace('[ ', '')
fottiti4volte = fottiti4volte.replace(']', '')
fottiti4volte = fottiti4volte.replace('  ', ' ')


iniziamoaragionare = fottiti4volte.split(' ')

chissasefunziona = []
for count in range(len(iniziamoaragionare)):
    value = float(iniziamoaragionare[count])
    chissasefunziona.append(value)
'''
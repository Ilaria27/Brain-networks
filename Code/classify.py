# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:20:34 2021

@author: Ilaria T
"""

import argparse
import os
import subprocess
import numpy as np
import pandas as pd
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('-m', help = 'Method', default = 'contrast-subgraph', choices = ['contrast-subgraph',
                                                                                     'GraphEmbs',
                                                                                     'BiomarkersSCHZ-master',
                                                                                     'graphclass-master'],
                                                                                        #'GroupINN',
                                                                                     type = str)
parser.add_argument('-d', help = 'Dataset', default = 'children', type = str)
parser.add_argument('-c1', help = 'Class 1 of dataset', default = 'asd', type = str)
parser.add_argument('-c2', help = 'Class 2 of dataset', default = 'td', type = str)
    
args = parser.parse_args()

subprocess.run('python preprocess.py -d {} -c1 {} -c2 {}'.format(args.d, args.c1, args.c2))

from preprocess import classes
print(classes)

#dir_m = "/{}".format(args.m)
if args.m == 'contrast-subgraph':
    alpha = input('Alpha parameter [0, 1] ')
    with open('method.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd {}\n".format(args.m))
        f.write("\npython2 cs.py {} {} {} {}\n".format(args.d, args.c1, args.c2, alpha))
        f.write("\npython2 cs.py {} {} {} {}\n".format(args.d, args.c2, args.c1, alpha))
        f.write("\npython cs_classification.py -d {} -c1 {} -c2 {}". format(args.d, args.c1, args.c2))
        f.write('sleep 30')
        f.close()
    subprocess.run("method.sh", shell = True)
    #importare matrici e pred
    os.chdir(args.m)
    X = pd.read_excel('features.xls', header=None, index_col= None, skiprows=1)
    for i in range(len(X)):
        for j in range(len(X.transpose())):        
            string = X.iloc[i,j]
            string = string.replace('\n', ' ')
            string = string.replace('  ', ' ')
            string = string.replace('    ', ' ')
            string = string.replace('   ', ' ')        
            string = string.replace('  ', ' ')
            string = string.replace('  ', ' ')
            string = string.replace('[ ', '')
            string = string.replace('[', '')
            string = string.replace(' ]', '')
            string = string.replace(']', '')
            string = string.split(' ')
            floats = []
            for v in string:
                floats.append(float(v))
            floats = np.array(floats)
            floats = floats.flatten()
            X.iloc[i,j] = floats
    print(X.iloc[0,0])
    X = np.array(X)
    print(X)
    Y = classes
elif args.m == 'GraphEmbs':
    os.chdir("{}/".format(args.m))
    subprocess.run('python run_synthetic_graphs.py {} {} {}'.format(args.d, args.c1, args.c2), shell = True)
    #features and predictions for classification
    X = np.loadtxt('results/asd/X/td_100_64_1024_0.1.txt')
    Y = np.loadtxt('results/asd/Y/td_100_64_1024_0.1.txt')
#elif args.m == 'GroupINN':
#    os.chdir("{}/".format(args.m))
#    subprocess.run('python groupinn.py {} {} {}'.format(args.d, args.c1, args.c2), shell = True)
elif args.m == 'BiomarkersSCHZ-master':
    os.chdir("{}/".format(args.m))
    with open('method.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("\npython2 main.py -dataset {} -c1 {} -c2 {} -connectivity Functional -resolution 83\n".format(args.d, args.c1, args.c2))
        f.write('sleep 60')
        f.close()
    subprocess.run("method.sh", shell = True)
    #accuracy values
    annots = loadmat('BiomarkersSCHZ-master/mat/abs_subcortical/GlobalResults_83_Functional_accuracy_children.mat')
    accs = annots['fc_83_mean_acc']
    print(accs)
elif args.m == 'graphclass-master':
    os.chdir("{}/".format(args.m))
    subprocess.run(' "C:/Programmi/R/R-4.0.2/bin/Rscript.exe" Try.R -d {} -c1 {} -c2 {}'.format(args.d, args.c1, args.c2), shell = True)
    #accuracy value
    acc = np.loadtxt('accuracy.txt')
    print(acc)
    
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import svm
#from sklearn.metrics import classification_report
import pandas as pd

if args.m == 'GraphEmbs':
        
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)
    
    svc = svm.SVC()
    parameters = {'C' : [1, 5, 10],
                  'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
                  'degree' : [2, 3, 4],
                  'gamma' : ('scale', 'auto')}
    k_f = KFold(5, shuffle=True, random_state=43).get_n_splits(X)
    grid_s = GridSearchCV(svc, parameters, refit = True, scoring = 'accuracy', cv = k_f)
    grid_s.fit(X, Y)
    y_true, y_pred = y_test, grid_s.predict(X_test)
    mean_acc = grid_s.cv_results_['mean_test_score'].mean()
    method_params = []
     
    results = pd.DataFrame([args.m, method_params, y_pred, y_true, mean_acc])
    results = pd.DataFrame(['GraphEmbs', method_params, y_pred, y_true, mean_acc])
    
    results = np.transpose(results)
    print(mean_acc)
elif args.m == 'contrast-subgraph':

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)
    
    svc = svm.SVC()
    parameters = {'C' : [1, 5, 10],
                  'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
                  'degree' : [2, 3, 4],
                  'gamma' : ('scale', 'auto')}
    k_f = KFold(5, shuffle=True, random_state=43).get_n_splits(X)
    grid_s = GridSearchCV(svc, parameters, refit = True, scoring = 'accuracy', cv = k_f)
    grid_s.fit(X, Y)
    y_true, y_pred = y_test, grid_s.predict(X_test)
    mean_acc = grid_s.cv_results_['mean_test_score'].mean()
    method_params = []
     
    results = pd.DataFrame([args.m, method_params, y_pred, y_true, mean_acc])
    results = pd.DataFrame(['GraphEmbs', method_params, y_pred, y_true, mean_acc])
    
    results = np.transpose(results)
    print(mean_acc)


'''
#save results
if args.m == 'GraphEmbs' or 'BiomarkersSCHZ-master' or 'graphclass-master':
    results.to_csv('../results.csv', header = None, index = None, mode = 'a', sep = ' ')
else:
    results.to_csv('results.csv', header = None, index = None, mode = 'a', sep = ' ')
'''

#print(classification_report(y_true, y_pred))

#group = parser.add_mutually_exclusive_group()
#group.add_argument("-v", "--verbose", action="store_true")
#group.add_argument("-q", "--quiet", action="store_true")
#parser.add_argument("x", type=int, help="the base")
#parser.add_argument("y", type=int, help="the exponent")
#args = parser.parse_args()
#answer = args.x**args.y

#if args.quiet:
#    print(answer)
#elif args.verbose:
#    print("{} to the power {} equals {}".format(args.x, args.y, answer))
#else:
#    print("{}^{} == {}".format(args.x, args.y, answer))
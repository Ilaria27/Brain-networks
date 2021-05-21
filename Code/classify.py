# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:20:34 2021

@author: Ilaria T
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-m', help = 'Method', default = 'contrast-subgraph', choices = ['contrast-subgraph',
                                                                                     'GraphEmbs',
                                                                                     'GroupINN',
                                                                                     'BiomarkersSCHZ-master'],
                                                                                     type = str)
parser.add_argument('-d', help = 'Dataset', default = 'children', type = str)
parser.add_argument('-c1', help = 'Class 1 of dataset', default = 'asd', type = str)
parser.add_argument('-c2', help = 'Class 2 of dataset', default = 'td', type = str)
    
args = parser.parse_args()

#dir_m = "/{}".format(args.m)
if args.m == 'contrast-subgraph':
    alpha = input('Alpha parameter [0, 1] ')
    with open('method.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd {}\n".format(args.m))
        f.write("\npython2 cs.py {} {} {} {}\n".format(args.d, args.c1, args.c2, alpha))
        f.write('sleep 30')
        f.close()
    subprocess.run("method.sh", shell = True)
elif args.m == 'GraphEmbs':
    os.chdir("{}/".format(args.m))
    subprocess.run('python run_synthetic_graphs.py {} {} {}'.format(args.d, args.c1, args.c2), shell = True)
    X = np.loadtxt('results/asd/X/td_100_64_1024_0.1.txt')
    Y = np.loadtxt('results/asd/Y/td_100_64_1024_0.1.txt')
elif args.m == 'GroupINN':
    os.chdir("{}/".format(args.m))
    subprocess.run('python groupinn.py {} {} {}'.format(args.d, args.c1, args.c2), shell = True)
elif args.m == 'BiomarkersSCHZ-master':
    os.chdir("{}/".format(args.m))
    with open('method.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("\npython2 main.py -dataset {} -c1 {} -c2 {} -connectivity Functional -resolution 83\n".format(args.d, args.c1, args.c2))
        f.write('sleep 60')
        f.close()
    subprocess.run("method.sh", shell = True)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=random_seed)

svc = svm.SVC()
parameters = {'C' : [1, 5, 10],
              'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
              'degree' : [2, 3, 4],
              'gamma' : ('scale', 'auto')}
k_f = KFold(5, shuffle=True, random_state=43).get_n_splits(X)
grid_s = GridSearchCV(svc, parameters, refit = True, scoring = 'accuracy', cv = k_f)
grid_s.fit(X, Y)
grid_s.predict(X_test)
grid_s.predict(X_test) == y_test
grid_s.cv_results_['mean_test_score']

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
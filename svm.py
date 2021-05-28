# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:42:40 2021

@author: Ilaria T
"""
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

random_seed = 42

X = np.loadtxt('GraphEmbs/results/asd/X/td_100_64_1024_0.1.txt')
Y = np.loadtxt('GraphEmbs/results/asd/Y/td_100_64_1024_0.1.txt')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=random_seed)

# Initialize SVM classifier
clf = svm.SVC()
# Fit data
clf = clf.fit(X_train, y_train)
# Predict the test set
predictions = clf.predict(X_test)
# Generate confusion matrix
matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show(matrix)
plt.show()

print(classification_report(y_test, predictions))

def CrossValidation(model, n_folds, x, y):
    k_f = KFold(n_folds, shuffle=True, random_state=43).get_n_splits(x)
    return(cross_val_score(model, x, y, scoring="accuracy", cv = k_f))

svc = svm.SVC()
parameters = {'C' : [1, 5, 10],
              'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
              'degree' : [2, 3, 4],
              'gamma' : ('scale', 'auto')}
k_f = KFold(5, shuffle=True, random_state=43).get_n_splits(X)
grid_s = GridSearchCV(svc, parameters, refit = True, scoring = 'accuracy', cv = k_f)
grid_s.fit(X_train, y_train)
grid_s.cv_results_
grid_s.cv_results_['mean_test_score']
grid_s.predict(X_test)
grid_s.predict(X_test) == y_test
best_parameters = grid_s.best_estimator_
best_parameters
type(best_parameters)
best_parameters.named_steps['fs']

svc = svm.SVC(C = 1, degree = 2, kernel = 'linear')
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
pred
y_test
svc.predict(X_test) == y_test
CrossValidation(SVM, 7, X, Y)

k_f = KFold(5, shuffle=True, random_state=43).get_n_splits(X)
grid_s = GridSearchCV(svc, parameters, refit = True, scoring = 'accuracy', cv = k_f)
grid_s.fit(X, Y)
grid_s.predict(X_test)
grid_s.predict(X_test) == y_test
grid_s.cv_results_['mean_test_score']


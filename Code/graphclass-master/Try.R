#!/usr/bin/env Rscript
library(devtools)
# install_github("jesusdaniel/graphclass")
# install.packages("penalizedSVM", repos='http://cran.us.r-project.org')
library(penalizedSVM)
library(graphclass)
#source("Classifiers/Cross-validation-function.R")
#source("Classifiers/train-test-functions-cv.R")

data(COBRE.data)
plot_adjmatrix(COBRE.data$X.cobre[1,])

X <- COBRE.data$X.cobre
Y <- COBRE.data$Y.cobre
Xnorm <- apply(X, 2, function(v) (v-mean(v))/sd(v))

#rho_seq <- c(10^(seq(2.5, -2, length.out = 31)))
#grid = data.frame(rbind(1e-4, rho_seq, 1e-5))
#parameters_grid <- lapply(grid, function(x) x)

NODES <- (1+sqrt(1+8*ncol(Xnorm)))/2
D <- construct_D(NODES)

#gc1_cv <- cross_validation_function(X = Xnorm, Y = factor(Y), 
#                                    parameters_grid = parameters_grid,
#                                    methodname = "gc1",
#                                    train_classifier = train_graphclass(X, Y, 
#                                                                        parameters_grid, 
#                                                                        list(D)), 
#                                   test_classifier = test_graphclass(train_graphclass(X, Y,
#                                                                                       parameters_grid,
#                                                                                       list(D)), X, Y),
#                                    folds = 10, algorithm_parameters = list(D),
#                                    parallel = T, num_clusters = 10, windows = T, 
#                                    nested_cv = F, save_files = F, filename = "", 
#                                    sparsity_results=T) 

#save(gc1_cv, file="res/gc1-cv.RData") 


fold_index <- (1:length(Y) %% 5) + 1
gclist <- list()
for(fold in 1:5) {
  foldout <- which(fold_index == fold)
  gclist[[fold]] <- graphclass(X = X[-foldout,], Y = Y[-foldout],
                               Xtest = X[foldout,], Ytest = Y[foldout],
                               type = "intersection",
                               lambda = 1e-4, rho = 1, gamma = 1e-5,
                               D = D)
}

which(fold_index == 5)
fold
# test error on each fold
lapply(gclist, function(gc) gc$test_error)
gclist[1]
Ypreds = lapply(gclist, function(gc) gc$Ypred)
Ypreds[[1]]
length(array(Ypreds[[2]]))
Ytest = Y[foldout]
acc = sum(array(Ypreds[[5]]) == Ytest)/length(Ytest)

write.table(acc, file="accuracy.txt", row.names = F, col.names = F)

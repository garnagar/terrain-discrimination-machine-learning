# To do for presentation
- classifier used and why this one
- accuracy and roc curve
- graph with comparison in Parameter(neurons) and accuracy
- different tried classifiers and why not these
- clustering graph

# Parameters to focus on
- activation functions (ReLU, ...)
- regularization
- convergence

# Dimensionality reduction
- linear transformation

# Resuts
(15,)
Average training time: 2.08087857008
Average score: 0.6404






introduction
Given Data set contains: 
400 samples
with 41125 data points each

0.36322569 0.36468692 0.36659293 ... 0.09645185 0.09093432 0.08933465
data point range between 0 and 1

number of samples for each class 100
4 classes

-> hidden layers in that range?
all classes
a) test 1 against 0

TODO
distribution ??

what is lbfgs ? why no learning rate or batch size

normalization ? L2 

is ReLu really the best ?

more hidden layers ?

conclusions !



future: 
dropout?
b) test against all classes
clustering











feedback
variances with activation functions

compare with nearest neighbors or quadratic classifier

validation loss and convergence loss plot

warm start






X0, y0 =separate_label(X,y,0)
X1, y1 =separate_label(X,y,1)
X2, y2 =separate_label(X,y,2)
X3, y3 =separate_label(X,y,3)
ex =[y0[0],y1[0], y2[0], y3[0]]
draw_terrain(get_patch(X,ex,dim=64),d=1)

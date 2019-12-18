import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score , cross_val_predict
from scipy.io import loadmat
from sklearn import linear_model.LogisticRegression as LR

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def get_av_scores_crossVal(classif):
    score = cross_val_score(classif, Adata, Alab, cv=5)
    return np.average(score)

def get_logistic_regression():
    lr = LR(C = 1, random_state= 1, )

mat = loadmat("terrain.mat", squeeze_me=True, struct_as_record=False)
X = np.array(mat['A'])/4.0 + 0.5
y = mat['y']

# choose class 0 and 1 to make it a 2 class problem
rng = np.random.randint(100)
X1 = separate_label(X,y,1)
X0 = separate_label(X,y,0)
X = np.concatenate((X1,X0))
y = np.concatenate((np.full(len(X1),1),np.full(len(X0),0)),axis=None)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=rng)

activations = [
'identity',
'logistic', # logistic sigmoid
'tanh', # hyperbolic tan function
'relu'] # rectified unit function

solvers = [
'lbfgs', # converges faster and performs better for small data sets
'sgd',  #
'adam'] # good for big data sets > 1000

alpha = [0.0001]

learning_rates = ['constant', 'invscaling', 'adaptive']

clf = MLPClassifier(hidden_layer_sizes=(2000,500,300), max_iter=1000, alpha=1e-4, solver='lbfgs', tol=1e-4, random_state=1,learning_rate_init=.005)

clf.fit(X_train, y_train)
score= clf.score(X_test,y_test)
print (score)

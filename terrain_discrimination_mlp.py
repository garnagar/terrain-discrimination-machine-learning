import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# FUNCTIONS
#################################################################

# Provided functions
#################################################################
def draw_terrain(Z,N=None,label="No label",d=1.7):

    if N is None:
        tam = len(Z)
    else:
        tam = 2**N+1

    A = np.max(abs(Z))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-d*A,d*A)


    X,Y = np.meshgrid( range(tam),range(tam) )

    #ax.plot_wireframe(X, Y, Z)
    ax.plot_surface(X, Y, Z)
    if np.max(np.abs(Z)) > 1e-5:
        ax.contour(X,Y,Z,dir='z',offset=-d*A)

    plt.title(label)
    plt.show()

def get_patch(X,i,dim=8):
    ptch = X[i].reshape(dim,dim)
    return ptch

def show_patch(X,i,y,dim=8):
    plt.gray()
    plt.matshow(get_patch(X,i,dim))
    plt.title("A sample digit: "+str(y[i]))
    plt.show()

# Custom functions
#################################################################
def load_data(file_name):
    """
    It loads set data from .mat file.

    file_name: path to the file
    return: set of data and labels
    """
    mat = loadmat(file_name, squeeze_me=True, struct_as_record=False)
    X = np.array(mat['A'])/4.0 + 0.5
    y = mat['y']
    return X, y

def separate_label(X,y,label):
    """
    It returns subset of data and labels with given label.

    X: array of data
    y: array of labels of the data
    return: array of data with array of labels
    """
    X_sep = np.array(np.zeros(len(X[0]))) #set width of row so vstack works
    y_sep = np.array([])
    for i in range(len(X)):
        if y[i] == label:
            X_sep = np.vstack((X_sep,X[i]))
            y_sep = np.append(y_sep,y[i])
    X_sep = np.delete(X_sep,0,axis=0) #delete first row
    return X_sep, y_sep

def separate_labels(X,y,labels):
    """
    It returns subset of data and labels with given labels.

    X: array of data
    y: array of labels of the data
    l: array of selected lables
    return: array of data and array of labels
    """
    X_sep = np.array(np.zeros(len(X[0]))) #set width of row so vstack works
    y_sep = np.array([])
    for i in range(len(X)):
        if np.isin(y[i],labels):
            X_sep = np.vstack((X_sep,X[i]))
            y_sep = np.append(y_sep,y[i])
    X_sep = np.delete(X_sep,0,axis=0) #delete first row
    return X_sep, y_sep

def classifier_test(X,y,clf,iter):
    avg_score = 0
    avg_time = 0
    for i in range(iter):
        # cross validation sets
        rng = np.random.RandomState(np.random.randint(100))
        X_train, X_test, y_train, y_test = train_test_split(X01, y01, test_size=.25, random_state=rng)
        # training
        t_ini = time()
        clf.fit(X_train, y_train)
        avg_time += time() - t_ini
        # scores
        avg_score += clf.score(X_test,y_test)
    avg_time /= iter
    avg_score /= iter
    print "Number of iterations: {}".format(iter)
    print "Average score: {}".format(avg_score)
    print "Average training time [s]: {}".format(avg_time)
    return avg_score, avg_time

def plot_score_time(x,x_label,yScore,yTime):
    plt.subplot(1,2,1)
    plt.plot(x,yScore,'--x')
    plt.xlabel(x_label)
    plt.ylabel("Average score")
    plt.subplot(1,2,2)
    plt.plot(x,yTime,'--x')
    plt.xlabel(x_label)
    plt.ylabel("Average training time [s]")
    plt.show()

# MAIN
#################################################################

# Init
#################################################################

# loading data
X, y = load_data("terrain.mat")

# sepating the cleses 0 and 1 from data set
X01, y01 = separate_labels(X,y,(0,1))

# Multy layer perceptron classifier
#################################################################
print "------MLP classifier"
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

# lbfgs solver
print "----lbfgs solver"

# activation func test
# activation_func = ['identity', 'logistic', 'tanh', 'relu']
# for a in activation_func:
#     print "--Activation function: {}".format(a)
#     mlp.set_params(activation=a)
#     s, t = classifier_test(X01,y01,mlp,20)
#     plt.plot(s, t, 'x',label=a)
# plt.legend(loc="lower right")
# plt.xlabel("Average score")
# plt.ylabel("Average training time [s]")
# plt.show()

# neuron number test
# neurons = [5,10,15,20,25]
# yTime = []
# yScore = []
# for n in neurons:
#     print "--Number of neurons: {}".format(n)
#     mlp.set_params(hidden_layer_sizes=(n,))
#     s, t = classifier_test(X01,y01,mlp,50)
#     yTime.append(t)
#     yScore.append(s)
#
# plot_score_time(neurons,"Number of neurons",yScore,yTime)
#
# mlp.set_params(hidden_layer_sizes=(15,)) # reset to default

# 2 layer neuron number test
neurons = [5,10,15,20,25]
for n1 in neurons:
    yTime = []
    yScore = []
    for n2 in neurons:
        print "--Number of neurons: ({},{})".format(n1,n2)
        mlp.set_params(hidden_layer_sizes=(n1,n2,))
        s, t = classifier_test(X01,y01,mlp,50)
        yTime.append(t)
        yScore.append(s)
    plt.subplot(1,2,1)
    plt.plot(neurons,yScore,'--x',label="{} N in layer 1".format(n1))
    plt.subplot(1,2,2)
    plt.plot(neurons,yTime,'--x',label="{} N in layer 1".format(n1))
plt.subplot(1,2,1)
plt.legend(loc="lower left")
plt.xlabel("Neurons in layer 2")
plt.ylabel("Average score")
plt.subplot(1,2,2)
plt.legend(loc="lower left")
plt.xlabel("Neurons in 2nd layer")
plt.ylabel("Average training time [s]")
plt.show()

mlp.set_params(hidden_layer_sizes=(15,)) # reset to default

# sgd solver
mlp.set_params(solver='sgd')
print "----sgd solver"

# learning rate test
# rate = [0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.03,0.04,0.05]
# yTime = []
# yScore = []
# for r in rate:
#     print "--Initial learning rate: {}".format(r)
#     mlp.set_params(learning_rate_init=r)
#     s, t = classifier_test(X01,y01,mlp,50)
#     yTime.append(t)
#     yScore.append(s)
#
# plot_score_time(rate,"Initial learning rate",yScore,yTime)
#
# mlp.set_params(learning_rate_init=0.001) # reset to default

# batch size test
# batch_size = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400]
# yTime = []
# yScore = []
# for b in batch_size:
#     print "--Batch size: {}".format(b)
#     mlp.set_params(batch_size=b)
#     s, t = classifier_test(X01,y01,mlp,20)
#     yTime.append(t)
#     yScore.append(s)
#
# plot_score_time(batch_size,"Batch size",yScore,yTime)
#
# mlp.set_params(batch_size='auto') # reset to default

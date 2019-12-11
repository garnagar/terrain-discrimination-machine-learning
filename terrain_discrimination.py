import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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
    RIt returns subset of data and labels with given labels.

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

# MAIN
#################################################################

#loading data
X, y = load_data("terrain.mat")

# sepating the cleses 0 and 1 from data set
X01, y01 = separate_labels(X,y,(0,1))
print X01
print y01

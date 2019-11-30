import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# FUNCTIONS
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

def show_patch(X,i,labels,dim=8):
    plt.gray()
    plt.matshow(get_patch(X,i,dim))
    plt.title("A sample digit: "+str(labels[i]))
    plt.show()

def separate_label(X,y,label):
    X_sep = []
    for i in range(len(X)):
        if y[i] == label:
            X_sep.append(X[i])
    return X_sep

# MAIN
#################################################################
mat = loadmat("terrain.mat", squeeze_me=True, struct_as_record=False)
X = np.array(mat['A'])/4.0 + 0.5
y = mat['y']

rng = np.random.randint(100)

X1 = separate_label(X,y,1)
X0 = separate_label(X,y,0)
X = np.concatenate((X1,X0))
y = np.concatenate((np.full(len(X1),1),np.full(len(X0),0)),axis=None)


clf = MLPClassifier(hidden_layer_sizes=(5,100), max_iter=1000, alpha=1e-4, solver='sgd', tol=1e-4, verbose=10, random_state=1, learning_rate_init=.005)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, random_state=rng)

clf.fit(X_train, y_train)
print clf.score(X_test,y_test)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.base import ClassifierMixin as Classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv1D, ConvLSTM2D, MaxPooling2D, UpSampling2D
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
    print(tam)
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
        #clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
        #for i in range(10):
        # clf.fit(X, y)
        clf.fit(X_train, y_train)
        avg_time += time() - t_ini
        # scores
        avg_score += clf.score(X_test,y_test)
    avg_time /= iter
    avg_score /= iter
    print ("Number of iterations: {}".format(itprint(yScore)er))
    print ("Average score: {}".format(avg_score))
    print ("Average training time [s]: {}".format(avg_time))
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

n_samples, n_features = X01.shape
n_digits = len(np.unique(y01))
labels = y01
# Perceptron classifier
#################################################################
def preceptron_():
    print ("------Perceptron classifier")
    per = Perceptron(verbose=0,max_iter=100000)
    classifier_test(X01,y01,per,10)


# k-Nearest Neigborgh classifier
#################################################################


# algorthm test
def knn_():
    print ("------k-NN classifier")
    knn = KNeighborsClassifier('auto')
    algorthms = ['auto','ball_tree','kd_tree','brute']
    neighbors = np.arange(1,20)
    for a in algorthms:
        yTime = []
        yScore = []
        print ("----{} algorithm".format(a))
        knn.set_params(algorithm=a)
        for n in neighbors:
            print ("--Number of neighbors: {}".format(n))
            knn.set_params(n_neighbors=n)
            s, t = classifier_test(X01,y01,knn,50)
            print(a)
            yTime.append(t)
            yScore.append(s)
        plt.subplot(1,2,1)
        plt.plot(neighbors,yScore,'--x',label="{}".format(a))
        plt.subplot(1,2,2)
        plt.plot(neighbors,yTime,'--x',label="{}".format(a))
    plt.subplot(1,2,1)
    plt.legend(loc="lower left")
    plt.xlabel("Neigborghs")
    plt.ylabel("Average score")
    plt.subplot(1,2,2)
    plt.legend(loc="lower left")
    plt.xlabel("Neigborghs")
    plt.ylabel("Average training time [s]")
    plt.show()


# Multy layer perceptron classifier
#################################################################

def mlp_():
    print ("------MLP classifier")
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10,), random_state=1)


    # data1, data2, label1, label2 = train_test_split(X01, y01, test_size=.20, random_state=0)
    # X_train = data1.reshape(data1.shape[0], data1.shape[1], 1)
    # Y_train = label1.reshape(label1.shape[0], 1)
    # print(X_train)
    # model = Sequential()
    # model.add(Conv1D(1, kernel_size=(5), activation='sigmoid', padding='same'))
    # model.add(Conv1D(1, kernel_size=(5), activation='sigmoid', padding='same'))
    # model.compile(loss='mse', optimizer='adam')
    #
    # model.fit(X_train, Y_train,
    #       batch_size=1, epochs=10, verbose=1)

    # lbfgs solver
    print ("----lbfgs solver")
    mlp.set_params(activation='tanh')

def show_pca_(model, x, y):
    components =[ n_digits,13,25, 50,75, len(y)//2]
    #scale the data
    scaled_data = StandardScaler().fit_transform(x)
    yTime = []
    yScore = []
    solver = ['auto', 'randomized']
    for sol in solver :
        yTime = []
        yScore = []
        labelVariance=0
        for c in components:
            pca= PCA(n_components = c, svd_solver = sol)
            reduced_data = pca.fit_transform(scaled_data)
            s, t = classifier_test(reduced_data, y01, model, 20)
            yTime.append(t)
            yScore.append(s)
            labelVariance= sum(pca.explained_variance_ratio_)
            print(labelVariance)
        plt.subplot(1,2,1)
        plt.plot(components,yScore,'--x',label="{}".format(sol))
        print(labelVariance)
        plt.subplot(1,2,2)
        plt.plot(components,yTime,'--x',label="{}".format(sol))
    plt.subplot(1,2,1)
    plt.legend(loc="lower left")
    plt.xlabel("Number of Components")
    plt.ylabel("Average score")
    plt.subplot(1,2,2)
    plt.legend(loc="lower left")
    plt.xlabel("Number of Components")
    plt.ylabel("Average training time [s]")
    plt.show()




#####################################
#visualize the data with the use of PCA
def visualize_(x_,y_, num = 4):
    scaled_data = StandardScaler().fit_transform(x_)
    #pca = KernelPCA(n_components = 3, kernel="poly", gamma=0.01 ,remove_zero_eig= True)
    pca = PCA(n_components = 3)

    reduced_data = pca.fit_transform(scaled_data)
    print(sum(pca.explained_variance_ratio_))

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    red_data_ =[[],[],[],[]]
    colors=['b','r','g','y']
    for i in range(0,num):
        red_data_[i] = reduced_data[y_==i]

        ax.scatter(red_data_[i][:, 0], red_data_[i][:, 1],
            zs=red_data_[i][:, 2], zdir='z', s=25, c=colors[i], depthshade=True)
    ax.set_title('Classes reduced to 3 dimensions for visualization')
    plt.grid()
    plt.show()



def show_kmeans_2():
    #pca to get 2 dimensions

    scaled_data = StandardScaler().fit_transform(X01)
    pca = PCA(n_components = 2)
    reduced_data = pca.fit_transform(scaled_data)
    print(pca.explained_variance_ratio_)
    kmeans = KMeans(n_clusters=n_digits)
    kmeans.fit(reduced_data)
    h=0.2
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    red_data_0 = reduced_data[y01==0]
    red_data_1 = reduced_data[y01==1]
    plt.plot(red_data_0[:, 0], red_data_0[:, 1],'k.', markersize=3, color = 'b')
    plt.plot(red_data_1[:, 0], red_data_1[:, 1],'k.', markersize=3, color = 'r')

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the PCA-reduced data (2classes)\n')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def kmeans_(X,y, n_digits, rng=1):
    avg_score= 0
    yScore =[]
    for k in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=rng)
        kmeans = KMeans(n_clusters=n_digits)
        kmeans.fit(X_train)
        predict = kmeans.predict(X_test)

        yScore.append([])
        for i in range(n_digits):
            yScore[k].append(sum(predict[y_test==i])/sum(predict))

        avg_score += sum(yScore[k])
    avg_score/= (k+1)
    print ("Number of iterations: {}".format(k))
    print ("Average score: {}".format(avg_score))

def show_kmeans_all(num = 4):
    #pca to get 2 dimensions
    scaled_data = StandardScaler().fit_transform(X)
    pca = PCA(n_components = 2)
    reduced_data = pca.fit_transform(scaled_data)
    print(pca.explained_variance_ratio_)
    kmeans = KMeans(n_clusters=num)
    kmeans.fit(reduced_data)
    h=0.2
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    red_data_ =[[],[],[],[]]
    colors=['b','r','g','y']
    for i in range(0,num):
        red_data_[i] = reduced_data[y==i]

        plt.plot(red_data_[i][:, 0], red_data_[i][:, 1],'k.', markersize=4, color = colors[i])
    #plt.plot(red_data_1[:, 0], red_data_[:, 1],'k.', markersize=3, color = 'r')

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the PCA-reduced data(4 classes)\n')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# activation func test
def activation_func_test(mlp):
    activation_func = ['identity', 'logistic', 'tanh', 'relu']
    for a in activation_func:
        print ("--Activation function: {}".format(a))
        mlp.set_params(activation=a)
        s, t = classifier_test(X01,y01,mlp,20)
        plt.plot(s, t, 'x',label=a)
    plt.legend(loc="lower right")
    plt.xlabel("Average score")
    plt.ylabel("Average training time [s]")
    plt.show()

# neuron number test
def neuron_num_test(mlp):
    neurons = [5,10,15,20,25]
    yTime = []
    yScore = []
    for n in neurons:
        print ("--Number of neurons: {}".format(n))
        mlp.set_params(hidden_layer_sizes=(n,))
        s, t = classifier_test(X01,y01,mlp,1)
        yTime.append(t)
        yScore.append(s)

    plot_score_time(neurons,"Number of neurons",yScore,yTime)

    mlp.set_params(hidden_layer_sizes=(15,)) # reset to default

    #2 layer neuron number test
    neurons = [5,10,15,20,25]
    for n1 in neurons:
        yTime = []
        yScore = []
        for n2 in neurons:
            print ("--Number of neurons: ({},{})".format(n1,n2))
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

# mlp.set_params(hidden_layer_sizes=(15,)) # reset to default

# sgd solver

def sgd_solver_test(mlp):
    print ("----sgd solver")
    mlp.set_params(solver='sgd')

    #learning rate test
    rate = [0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.03,0.04,0.05]
    yTime = []
    yScore = []
    for r in rate:
        print ("--Initial learning rate: {}".format(r))
        mlp.set_params(learning_rate_init=r)
        s, t = classifier_test(X01,y01,mlp,50)
        yTime.append(t)
        yScore.append(s)

    plot_score_time(rate,"Initial learning rate",yScore,yTime)

    mlp.set_params(learning_rate_init=0.001) # reset to default

    #batch size test
    batch_size = [5,10,20,40,50,100,200]
    yTime = []
    yScore = []
    for b in batch_size:
        print ("--Batch size: {}".format(b))
        mlp.set_params(batch_size=b)
        s, t = classifier_test(X01,y01,mlp,20)
        yTime.append(t)
        yScore.append(s)

    plot_score_time(batch_size,"Batch size",yScore,yTime)

    #mlp.set_params(batch_size='auto') # reset to default








mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20,), random_state=1, learning_rate_init = 0.001, activation = "relu", batch_size = 25)
#knn_()
show_pca_(mlp, X, y)
#s,t = classifier_test(X, y, mlp, 20)
#visualize_(X01, y01, 2)
#visualize_(X,y, 4)
#show_kmeans_2()
#show_kmeans_all()
#kmeans_(X,y,4)
#kmeans_(X01,y01,2)

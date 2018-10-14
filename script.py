import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from math import sqrt
import pickle
from pylab import *
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cls_pred, cls_true):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    #cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    w = initialWeights.reshape((n_feature+1,1))   # weights
    n_data_O = np.ones((n_data,1))
    x = np.hstack((n_data_O,train_data))
    x_w = np.dot(x,w)
    y = sigmoid(x_w)

    error1 = labeli*np.log(y)
    error2 = 1.0 - labeli
    error3 = np.log(1.0-y)
    error4 = error2*error3
    error5 = error1 + error4
    error = -np.sum(error5)
    error = error/n_data
    y_label_i = y-labeli
    y_label_i = y_label_i.reshape(n_data,1)
    
    error_grad = y_label_i * x
    error_grad = np.sum(error_grad, axis=0)
    error_grad = error_grad/n_data
    print("blr error:"+str(error))
    #print("blr error_grad:"+error_grad)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    #label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0];
    label = np.zeros((data.shape[0], 1))
    n_feature = data.shape[1];
    label = np.zeros((n_data,1));
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    n_data_1 = np.ones((n_data, 1))
    x = np.hstack((n_data_1,data))

    xW = np.dot(x, W)
    label1 = sigmoid(xW)   # compute probabilities
    label2 = np.argmax(label1, axis=1)    # get maximum for each class
    label = label2.reshape((n_data,1))
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    w = params.reshape((n_feature+1,10))   # weights
    n_data_O = np.ones((n_data,1))
    x = np.hstack((n_data_O,train_data))
    y = softmax(w,x) # change? to softmax?
    
    
    error = -np.sum(np.sum(labeli*np.log(y)))
    error = error/n_data
    y_l = y-labeli
    e1 = np.dot(x.T,y_l)
    error_grad = e1.flatten()/n_data
    error_grad = error_grad
    print("error: "+str(error))
    #print("error_grad: "+error_grad)
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    n_data = data.shape[0];
    n_data_1 = np.ones((n_data, 1))
    x = np.hstack((n_data_1,data))

    #xW = np.dot(x, W)
    label1 = softmax(W,x)   # compute probabilities
    label2 = np.argmax(label1, axis=1)    # get maximum for each class
    label = label2.reshape((n_data,1))
    return label

def softmax(W,x):
    vec = np.dot(x,W);
    vec1 = np.exp(vec);
    vecd = np.sum(vec1,axis=1);
    vecd = vecd.reshape(vecd.shape[0],1)
    res = vec1/vecd;
    return res;


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
plot_confusion_matrix(predicted_label, train_label)

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
plot_confusion_matrix(predicted_label, test_label)

# """
# Script for Support Vector Machine
# """

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
#Linear kernel
print('SVM with linear kernel')
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')


# Radial basis function with gamma = 1
print('\n\n SVM with radial basis function, gamma = 1')
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')


# Radial basis function with gamma = 0
print('\n\n SVM with radial basis function, gamma = 0')
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')


# Radial basis function with C = 1, 10, 20 ... 100
print('\n\n SVM with radial basis function, different values of C')
train_accuracy = np.zeros(11)
valid_accuracy = np.zeros(11)
test_accuracy = np.zeros(11)
C_val = np.zeros(11)
C_val[0] = 1.0   # first value is 1
C_val[1:] = [x for x in np.arange(10.0, 101.0, 10.0)]    # rest is 10, 20 ... 100
### for every C, train and compute accuracy ###
for i in range(11):
    clf = SVC(C=C_val[i],kernel='rbf')
    clf.fit(train_data, train_label.flatten())
    train_accuracy[i] = 100*clf.score(train_data, train_label)
    valid_accuracy[i] = 100*clf.score(validation_data, validation_label)
    test_accuracy[i] = 100*clf.score(test_data, test_label)

pickle.dump((train_accuracy, valid_accuracy, test_accuracy),open("rbf_cval.pickle","wb"))
#(train_accuracy, valid_accuracy, test_accuracy) = pickle.load(open('rbf_cval.pickle','rb'))

# Plot accuracy of SVM with rbf vs C values
plot(C_val, train_accuracy, 'o-',
    C_val, valid_accuracy,'o-',
    C_val, test_accuracy, 'o-')

xlabel('C values')
ylabel('Accuracy (%)')
title('Accuracy using SVM with Gaussian kernel and different values of C')
legend(('Training','Validation','Test'), loc='lower right')
grid(True)
tight_layout()
savefig("rbf_cval.png")
show()

"""
Script for Extra Credit Part
"""
FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
plot_confusion_matrix(predicted_label_b, train_label)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
plot_confusion_matrix(predicted_label_b, test_label)

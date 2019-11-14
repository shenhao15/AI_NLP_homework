from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def sigmoid(z):
    '''
    Compute the sigmoid of z
    Arguments: z -- a scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    '''
    s = 1. / (np.exp(-1 * z) + 1)

    return s


# Random innitialize the parameters

def initialize_parameters(dim):
    '''
    Argument: dim -- size of the w vector

    Returns:
    w -- initialized vector of shape (dim,1)
    b -- initializaed scalar
    '''

    w = np.random.randn(dim, 1)
    b = np.random.ranf()

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    '''
    Implement the cost function and its gradient for the propagation
    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    '''
    m = X.shape[1]
    A = sigmoid(np.dot(X, w) + b)

    ## adjust value to avoid log0 and log1

    A_2 = A
    for i in range(A_2.shape[0]):
        if A_2[i][0] == 1:
            A[i][0] = 0.999999999
        elif A_2[i][0] == 0:
            A[i][0] = 1e-10
    cost = (-1. / m) * np.sum(np.dot(np.log(A), Y.T) + np.dot(np.log(1. - A), 1. - Y.T))

    dw = (1 / m) * np.dot(X.T, np.subtract(A, Y))
    db = (1 / m) * np.sum(np.subtract(A, Y))

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw,
             'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''
    This function optimize w and b by running a gradient descen algorithm

    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params - dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    '''

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 10 == 0:
            costs.append(cost)
        if print_cost and i % 10 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights
    b -- bias
    X -- data

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    n = X.shape[0]
    Y_prediction = np.zeros((1, n))
    w = w.reshape(m, 1)

    A = sigmoid(np.dot(X, w) + b)

    for i in range(A.shape[1]):
        if A[i][0] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

            # assert(Y_prediction.shape == (1,m))

    return Y_prediction


def compute_prediction_accuracy(predicted_value, actual_value):
    total_record_number = actual_value.shape[0]
    correct_number = 0
    for i in range(total_record_number):
        if actual_value[i][0] == predicted_value[i][0]:
            correct_number += 1

    return correct_number / total_record_number


def model(X_train, Y_trein, X_test, Y_test, num_iterations, learning_rate, print_cost):
    """
    Build the logistic regression model by calling all the functions you have implemented.
    Arguments:
    X_train - training set
    Y_train - training label
    X_test - test set
    Y_test - test label
    num_iteration - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d - dictionary should contain following information w,b,training_accuracy, test_accuracy,cost
    eg: d = {"w":w,
             "b":b,
             "training_accuracy": traing_accuracy,
             "test_accuracy":test_accuracy,
             "cost":cost}
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    init_w, init_b = initialize_parameters(m)
    last_param = optimize(init_w, init_b, X_train, y_train.reshape(n, -1), num_iterations, learning_rate, print_cost)
    trained_weights = last_param[0]['w']
    trained_b = last_param[0]['b']
    cost = last_param[2]
    training_predicted_value = predict(trained_weights, trained_b, X_train).T
    training_accuracy = compute_prediction_accuracy(training_predicted_value, y_train.reshape(-1, 1))
    testset_predicted_value = predict(trained_weights, trained_b, X_test).T
    test_accuracy = compute_prediction_accuracy(testset_predicted_value, y_test.reshape(-1, 1))

    d = {"w": trained_weights, "b": trained_b, "training_accuracy": training_accuracy, "test_accuracy": test_accuracy,
         "cost": cost}

    return d


# Loading the data
digits = datasets.load_digits()

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)


# reformulate the label.
# If the digit is smaller than 5, the label is 0.
# If the digit is larger than 5, the label is 1.

y_train[y_train < 5 ] = 0
y_train[y_train >= 5] = 1
y_test[y_test < 5] = 0
y_test[y_test >= 5] = 1

# Vilizating the data
plt.figure(1)
for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(digits.data[i-1].reshape([8,8]),cmap=plt.cm.gray_r)
    plt.text(3,10,str(digits.target[i-1]))
    plt.xticks([])
    plt.yticks([])


#learning curve with different learning rate
num_iterations = 2000
print_cost = False

train_acc = {}
test_acc = {}
for learning_rate in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5]:
    pred = model(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1),
                             num_iterations, learning_rate, print_cost = False)
    train_acc[learning_rate] = pred["training_accuracy"]
    test_acc[learning_rate] = pred["test_accuracy"]


plt.figure(2)
plt.title('learning_rate - accuracy')
plt.plot(list(train_acc.values()), c='red',  label = 'training_accuracy')
plt.plot(list(test_acc.values()), c='green',  label = 'test_accuracy')
plt.xticks(range(len(test_acc)), list(test_acc.keys()))
plt.xlabel('learning_rate')
plt.ylabel('accuracy')
plt.legend(loc='lower left')


#learning curve with different num_iterations
learning_rate = 1e-3

train_acc = {}
test_acc = {}
for num_iterations in range(100, 5000, 500):
    pred = model(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1), num_iterations, learning_rate, print_cost = False)
    train_acc[num_iterations] = pred["training_accuracy"]
    test_acc[num_iterations] = pred["test_accuracy"]



plt.figure(3)
plt.title('iterations - accuracy')
plt.plot(list(train_acc.values()), c='red',  label = 'training_accuracy')
plt.plot(list(test_acc.values()), c='blue',  label = 'test_accuracy')
plt.xticks(range(len(test_acc)), list(test_acc.keys()))
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend(loc='lower left')

plt.show()

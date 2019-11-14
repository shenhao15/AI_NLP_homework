from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from icecream import ic

# define active function定义激活函数
def softmax(X):  # softmax函数
    #解决当输入一个较大的数值时，sofmax函数上溢问题
    X -= np.max(X)
    z = np.exp(X) / np.sum(np.exp(X))
    return z

# innitialize the parameters
def initialize_parameters(dim_x):

    w = np.zeros((dim_x, 10))

    return w

# Forward backward propagation
def propagate(w, lenda, X, Y):

    m = X.shape[1]

    A = softmax(np.dot(w.T, X))

    # 解决divide by zero encountered in log问题
    epsilon = 1e-5
    cost = -1/m * np.sum(Y * np.log(A + epsilon)) + 0.5 * lenda * np.sum(w * w)
    dw = -1 / m * np.dot(X, (Y - A).T) + lenda * w

    assert (dw.shape == w.shape)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = dw

    return grads, cost

def optimize(w, lenda, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, lenda, X, Y)

        w = w - learning_rate * grads

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return w, grads, costs


# 预测
# Calculate  Ŷ = softmax(wT∗X)
def predict(w, X):
    m = X.shape[1]
    Y_prediction = np.zeros((10, m))
    w = w.reshape(X.shape[0], 10)

    Y_prediction = softmax(np.dot(w.T, X))

    assert (Y_prediction.shape == (10, m))

    return Y_prediction


def model(lenda, X_train, Y_train, X_test, Y_test, num_iterations,
          learning_rate, print_cost=False):

    dim = X_train.shape[0]

    w_temp = initialize_parameters(dim)
    w, grads, cost = optimize(w_temp, lenda, X_train, Y_train,
                                    num_iterations, learning_rate,
                                    print_cost)

    # 预测结果0-9取概率最大项即为所预测数字，计算accuracy
    pred_train = predict(w, X_train)
    training_accuracy = (np.sum(np.argmax(pred_train, axis=0).reshape(1,-1)
                                == np.argmax(Y_train, axis=0).reshape(1,-1))) / Y_train.shape[1]
    pred_test = predict(w, X_test)
    test_accuracy = (np.sum(np.argmax(pred_test, axis=0) ==
                            np.argmax(Y_test, axis=0))) / Y_test.shape[1]

    d = {"w": w,
         "training_accuracy": training_accuracy,
         "test_accuracy": test_accuracy,
         "cost": cost}

    return d


# Loading the data
digits = datasets.load_digits()

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)

# Vilizating the data
plt.figure(1)
for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(digits.data[i-1].reshape([8,8]),cmap=plt.cm.gray_r)
    plt.text(3,10,str(digits.target[i-1]))
    plt.xticks([])
    plt.yticks([])


# 数据处理
X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

#one-hot ylable
train = np.zeros((10, y_train.shape[1]))
for i in range(y_train.shape[1]):
    train[y_train[0,i], i] = 1
test = np.zeros((10, y_test.shape[1]))
for j in range(y_test.shape[1]):
    test[y_test[0,j], j] = 1


#Normalize data 正则化
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)


#learning curve with different learning rate
num_iterations =1000
print_cost = False
lenda = 1e-8

train_acc = {}
test_acc = {}

for learning_rate in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5]:
    prd = model(lenda, X_train, train, X_test, test,
                num_iterations, learning_rate, print_cost)
    train_acc[learning_rate] = prd["training_accuracy"]
    test_acc[learning_rate] = prd["test_accuracy"]

plt.figure(2)
plt.title('learning_rate - accuracy')
plt.plot(list(train_acc.values()), c='red',  label = 'training_accuracy')
plt.plot(list(test_acc.values()), c='blue',  label = 'test_accuracy')
#plt.plot(cost.values(), c='green',  label = 'cost')
plt.xticks(range(len(test_acc)), list(test_acc.keys()))
plt.xlabel('learning_rate')
plt.ylabel('accuracy')
plt.legend(loc='lower left')


#learning curve with different num_iterations
learning_rate = 1e-3

train_acc = {}
test_acc = {}

for num_iterations in range(100, 5000, 100):
    prd = model(lenda, X_train, train, X_test, test,
                num_iterations, learning_rate, print_cost)
    train_acc[num_iterations] = prd["training_accuracy"]
    test_acc[num_iterations] = prd["test_accuracy"]

plt.figure(3)
plt.title('iterations - accuracy')
plt.plot(list(train_acc.values()), c='red',  label = 'training_accuracy')
plt.plot(list(test_acc.values()), c='blue',  label = 'test_accuracy')
#plt.plot(cost.values(), c='green',  label = 'cost')
plt.xticks(range(len(test_acc)), list(test_acc.keys()))
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend(loc='lower left')

plt.show()

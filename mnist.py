import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Parameters
epochs = 100
lr = .1

def vectorize_sequence(X,num_classes):
    vectorize_vector = np.zeros((X.shape[0],num_classes))
    for index,x in enumerate(X):
        vectorize_vector[index,x] = 1
    return vectorize_vector


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train, (-1, 28 * 28))
x_train = np.array(x_train)/255
y_train = vectorize_sequence(y_train, 10)

x_test = np.reshape(x_test, (-1, 28 * 28))
x_test = np.array(x_test)/255
y_test = vectorize_sequence(y_test, 10)

#Show image
#plt.imshow(x_train[0].reshape(28,28),cmap='gray')
#plt.show()

X = x_train[0:1000,:]
Y = y_train[0:1000,:]

X = X.T
Y = Y.T

#Parameters
W0 = np.random.randn(512,784)*0.01
B0 = np.random.randn(512,1)*0.01
W1 = np.random.randn(10,512)*0.01
B1 = np.random.randn(10,1)*0.01

def relu(X):
    return np.maximum(X,0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sign(X):
    X[X >= 0]= 1
    X[X < 0] = 0
    return X

for i in range(epochs):
    #Forward Pass
    Z0 = np.matmul(W0,X) + B0
    A0 = relu(Z0)
    Z1 = np.matmul(W1,A0) + B1
    # print(Z1)
    A1 = sigmoid(Z1)

    J = np.sum(np.square(A1 - Y))
    print(J)

    #Backpropagation
    dA1 = A1 - Y
    dZ1 = dA1*A1*(1-A1)
    dW1 = np.matmul(dZ1,A0.T)/X.shape[1]
    dB1 = np.sum(dZ1, axis=1, keepdims=True)/X.shape[1]
    dA0 = np.matmul(W1.T,dZ1)
    dZ0 = dA0 * sign(Z0)
    dW0 = np.matmul(dZ0,X.T)/X.shape[1]
    dB0 = np.sum(dZ0, axis=1, keepdims= True)/X.shape[1]


    #Update weights and biases
    W0 -= lr * dW0
    B0 -= lr * dB0
    W1 -= lr * dW1
    B1 -= lr * dB1

X = x_test[0:5,:]
X = X.T

Z0 = np.matmul(W0,X) + B0
A0 = relu(Z0)
Z1 = np.matmul(W1,A0) + B1
A1 = sigmoid(Z1)

print(np.argmax(A1,axis=0))
print(np.argmax(y_test[0:5],axis=1))
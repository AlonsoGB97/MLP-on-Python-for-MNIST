import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Parameters
epochs = 100
lr = .001
batch_size = 128

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

num_samples = x_train.shape[0]
iterations = np.ceil(num_samples/batch_size).astype(int)

#Show image
#plt.imshow(x_train[0].reshape(28,28),cmap='gray')
#plt.show()

X = x_train[0:2000,:]
Y = y_train[0:2000,:]

X = X.T
Y = Y.T

#Parameters
W1 = np.random.randn(512,784)*0.01
B1 = np.random.randn(512,1)*0.01
W2 = np.random.randn(10,512)*0.01
B2 = np.random.randn(10,1)*0.01

def relu(X):
    return np.maximum(X,0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sign(X):
    X[X >= 0]= 1
    X[X < 0] = 0
    return X

for i in range(epochs):
    #Set gradients to 0
    dW2 = 0
    dB2 = 0
    dW1 = 0
    dB1 = 0
    J = 0
    for j in range(iterations):
        A0 = X[:,j * batch_size : (j + 1) * batch_size]
        Y_batch = Y[:,j * batch_size : (j + 1) * batch_size]

        #Forward Pass
        Z1 = np.matmul(W1,A0) + B1
        A1 = relu(Z1)
        Z2 = np.matmul(W2,A1) + B2
        A2 = sigmoid(Z2)
        J += np.sum(np.square(A2 - Y_batch))

        #Backpropagation
        dA2 = A2 - Y_batch
        dZ2 = dA2*A2*(1-A2)
        dW2 += np.matmul(dZ2,A1.T)
        dB2 += np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.matmul(W2.T,dZ2)
        dZ1 = dA1 * sign(Z1)
        dW1 += np.matmul(dZ1,A0.T)
        dB1 += np.sum(dZ1, axis=1, keepdims= True)

    #Update weights and biases
    W1 -= lr * dW1
    B1 -= lr * dB1
    W2 -= lr * dW2
    B2 -= lr * dB2
    J /= 2000
    print("Loss at epoch {0}: {1}".format(i,J))


#Accuracy
x_test = x_test.T

Z1 = np.matmul(W1,x_test) + B1
A1 = relu(Z1)
Z2 = np.matmul(W2,A1) + B2
A2 = sigmoid(Z2)

y_pred = np.argmax(A2,axis=0)
y_true = np.argmax(y_test,axis=1)

values = np.equal(y_pred,y_true)
print(np.sum(values)/y_pred.shape[0])
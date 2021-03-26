import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import time

# This dictionary defines the colormap
cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1
        'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1
        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }
GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

# 3 independent variables
input_data = np.array([[0,1,0],
                      [0,0,1],
                      [1,0,0],
                      [1,1,0],
                      [1,1,1],
                      [0,1,1],
                      [0,1,0]])

labels = np.array([[1, 0, 0, 1, 1, 0, 1]]).reshape(7,1)

# Initial Parameters
neuron_weights = np.random.rand(3,1)
bias = np.random.rand(1) # needed to add to the sumproduct of neurons to determine a threshold
learning_rate = 0.05

# activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))   

# store weights + error to view training
epochs = 500
w1 = [0]*epochs
w2 = [0]*epochs
w3 = [0]*epochs
err = [0]*epochs

# Training session
for epoch in range(epochs):

    # Forward feeding process
    inputs = input_data
    XW = np.dot(inputs, neuron_weights) + bias # w1.x1 + w2.x2 .... for all neurons
    pred = sigmoid(XW) # returns the result to within [0,1] (because of labels)
    error = pred - labels # find error
    
    #print("Error at epoch "+str(epoch)+": "+str(round(error.sum(), 4)))
    w1[epoch] = neuron_weights[0][0]
    w2[epoch] = neuron_weights[1][0]
    w3[epoch] = neuron_weights[2][0] 
    err[epoch] = error.sum()

    # Backpropagation
    d_error_d_pred = error # change of error w.r.t. pred
    d_pred_d_weight = sigmoid_derivative(pred) # derivative of sigmoid(x) w.r.t. x
    d_cost_d_weight = d_error_d_pred * d_pred_d_weight # multiplies the slope by the error terms
    
    inputs = input_data.T
    neuron_weights = neuron_weights - learning_rate*np.dot(inputs, d_cost_d_weight) # Gradient descent
    
    for num in d_cost_d_weight:
        bias = bias - learning_rate*num

# Plotting weights to view training convergence
ax = plt.axes(projection='3d')
ax.scatter3D(w1, w2, w3, c=err, cmap=GnRd)
plt.title('3 Neuron Weights over '+str(epochs)+' epochs, coloured by Error')
plt.show()

# Simple error overtime
plt.plot(err)
plt.title('Error over progression over '+str(epochs)+' epochs.')
plt.show()

# Make Predictions (should be near 0)
test_input = np.array([1,0,0])
net_output = np.dot(test_input, neuron_weights) + bias
result = sigmoid(net_output)

# Make Predictions (should be near 1)
test_input = np.array([0,1,0])
net_output = np.dot(test_input, neuron_weights) + bias
result = sigmoid(net_output)



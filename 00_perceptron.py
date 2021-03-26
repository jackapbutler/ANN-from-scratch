import numpy as np

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
np.random.seed(3)
neuron_weights = np.random.rand(3,1)
print(neuron_weights)
bias = np.random.rand(1) # needed to add to the sumproduct of neurons
learning_rate = 0.05

# functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

# Training session for Basic ANN
for epoch in range(100):

    # Forward feeding process
    inputs = input_data
    XW = np.dot(inputs, neuron_weights) + bias # w1.x1 + w2.x2 .... for all neurons
    output = sigmoid(XW) # returns the result to within [0,1] (because of labels)
    error = output - labels # find error
    #print("Error at epoch "+str(epoch)+": "+str(round(error.sum(), 4)))

    # Backpropagation
    cost = error # cost function can be changed
    dpred = sigmoid_derivative(output) # finds the slope at each network output point
    z_del = cost * dpred # multiplies the slope by the error terms
    inputs = input_data.T
    neuron_weights = neuron_weights - learning_rate*np.dot(inputs, z_del) # Gradient descent
    
    for num in z_del:
        bias = bias - learning_rate*num

# Make Predictions (should be near 0)
test_input = np.array([1,0,0])
net_output = np.dot(test_input, neuron_weights) + bias
result = sigmoid(net_output)

# Make Predictions (should be near 1)
test_input = np.array([0,1,0])
net_output = np.dot(test_input, neuron_weights) + bias
result = sigmoid(net_output)



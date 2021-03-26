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
neuron_weights = np.random.rand(3,1)
bias = np.random.rand(1) # needed to add to the sumproduct of neurons
learning_rate = 0.05

# functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

# Training session for Basic ANN
for epoch in range(10000):

    # Forward feeding process
    inputs = input_data
    XW = np.dot(inputs, neuron_weights) + bias # w1.x1 + w2.x2 .... for all neurons
    z = sigmoid(XW) # returns the result to within [0,1] (because of labels)
    error = z - labels # find error
    print(error.sum())

    # Backpropagation
    dpred = sigmoid_derivative(z)
    z_del = error * dpred
    inputs = input_data.T
    neuron_weights = neuron_weights - learning_rate*np.dot(inputs, z_del)
    
    for num in z_del:
        bias = bias - learning_rate*num


print("Final error on each sample")
print(error)

# Make Predictions (should be near 0)
test_input = np.array([1,0,0])
net_output = np.dot(test_input, neuron_weights) + bias
result = sigmoid(net_output)
print(result)

# Make Predictions (should be near 1)
test_input = np.array([0,1,0])
net_output = np.dot(test_input, neuron_weights) + bias
result = sigmoid(net_output)
print(result)



import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None


# First, you'll need to initialize the weights. We want these to be small such 
# that the input to the sigmoid is in the linear region near 0 and not squashed 
# at the high and low ends. It's also important to initialize them randomly so that 
# they all have different starting values and diverge, breaking symmetry. So, 
# we'll initialize the weights from a normal distribution centered at 0. A good 
# value for the scale is 1/√n where n is the number of input units. This keeps 
# the input to the sigmoid low for increasing numbers of input units.

# Initialize weights
weights = np.random.normal(scale=1/n_features**.5, size=n_features)

# Neural Network parameters
epochs = 1000
learnrate = 0.5

# Steps:
# 1) set the weight step to zero: delta_w = 0
# 2) for each record in the training data:
#     Make a forward pass through network (calculate y_hat = activation fucntion)
#     calculate the error gradient in the output unit: error_grad = (y-y_hat)*f'(h)*x
#     update the weights: w = w + learningrate * delta_w/m (m is the numb of records so that we can reduce large 
#         variations in the training data)
#     repeate for e epochs

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Activation of the output unit
        output = sigmoid(np.dot(x, weights))

        # The error, the target minues the network output
        error = y - output

        # The gradient descent step, the error times the gradient times the inputs
        # Error * derivative of the activation function * input
        del_w += error * output * (1 - output) * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
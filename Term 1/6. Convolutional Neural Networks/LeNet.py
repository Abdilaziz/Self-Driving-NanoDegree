# Load Data Set
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# The Standard LeNet Architecture accepts 32x32xC so lets adjust the default images from 28x28x1
import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

# Preprocess Data. Suffle the training data so it can be randomly sorted for Stochastic Gradient Descent when taking a batch.

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)




import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128




from tensorflow.contrib.layers import flatten

# Specs
# Convolution layer 1. The output shape should be 28x28x6.
# Activation 1. Your choice of activation function.
# Pooling layer 1. The output shape should be 14x14x6.
# Convolution layer 2. The output shape should be 10x10x16.
# Activation 2. Your choice of activation function.
# Poling layer 2. The output shape should be 5x5x16.
# Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
# Fully connected layer 1. This should have 120 outputs.
# Activation 3. Your choice of activation function.
# Fully connected layer 2. This should have 84 outputs.
# Activation 4. Your choice of activation function.
# Fully connected layer 3. This should have 10 outputs.
# You'll return the result of the final fully connected layer from the LeNet function.


def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu,stddev=sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16),mean=mu,stddev=sigma)),
    'wfc1': tf.Variable(tf.truncated_normal(shape=(5*5*16,120), mean=mu, stddev=sigma)),
    'wfc2': tf.Variable(tf.truncated_normal(shape=(120,84) , mean=mu , stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(84,10), mean=mu,stddev=sigma ))}

    biases = {
    'bc1': tf.Variable(tf.zeros((6))),
    'bc2': tf.Variable(tf.zeros((16))),
    'bfc1': tf.Variable(tf.zeros((120))),
    'bfc2': tf.Variable(tf.zeros(84)),
    'out': tf.Variable(tf.zeros((10)))}

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6. (Change in width and height comes from using VALID padding)
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc1']
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # TODO: Layer 2: Convolutional. Input = 14x14x6  Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1,weights['wc2'],strides=[1,1,1,1],padding='VALID') + biases['bc2']
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(conv2,weights['wfc1']),biases['bfc1'])

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1,weights['wfc2']),biases['bfc2'])

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    
    return logits




x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


# Training Pipline
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
# The tf.train.AdamOptimizer uses Kingma and Ba's Adam algorithm to control the 
# learning rate. Adam offers several advantages over the simple 
# tf.train.GradientDescentOptimizer. Foremost is that it uses moving 
# averages of the parameters (momentum); Bengio discusses the reasons for why 
# this is beneficial in Section 3.1.1 of this paper. Simply put, this enables Adam 
# to use a larger effective step size, and the algorithm will converge to this step 
# size without fine tuning.

# The main down side of the algorithm is that Adam requires more computation to be performed for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient); and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter). A simple tf.train.GradientDescentOptimizer could equally be used in your MLP, but would require more hyperparameter tuning before it would converge as quickly.
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# Model Evalution

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


    # Train the Model

    with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train) # Suffule for stochastic gradient disent
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

# evaluate the Model

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
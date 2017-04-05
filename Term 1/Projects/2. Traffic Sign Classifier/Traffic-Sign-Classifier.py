# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p' # if there wasnt a validation set, you would remove a percent of the training set 
                            # it could be done with from sklearn.model_selection import train_test_split (usually 20%)
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# The pickled data is a dictionary with 4 key/value pairs:

# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2)
# representing coordinates of a bounding box around the sign in the image.

# **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS 
# RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

# Number of training examples
n_train = X_train.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# shape of an traffic sign image
image_shape = X_train[0].shape

# Numb of unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# % of samples per class


# Visualization of the dataset
import matplotlib.pyplot as plt
import random

# Example Images
# index = random.randint(0,n_train)
# image = X_train[index].squeeze()
# print(y_train[index])
# plt.imshow(image)
# plt.show()

# Count of each sign
# plt.hist(y_train,1,normed=1, facecolor='green', alpha=0.75)
# plt.show()



# Design and Test a Model Architecture

# Pre-Process Data Set
# Techniques: Normalization, rgb to grayscale,etc.
import numpy as np


# RGB to Gray
def rgb2Gray(image):
    grey = np.zeros((image.shape[0], image.shape[1], 1)) # init 2D numpy array
    grey[:,:,0] = (0.2125*image[:,:,0] + 0.7154*image[:,:,1] + 0.0721*image[:,:,2])
    return grey

def normalization(image):
    image = (image - np.mean(image))/np.std(image)
    return image

def preprocess(image):
    image = rgb2Gray(image)
    image = normalization(image)
    return image

# def rgb2Gray4D(X_train):
#     X_grey_train = np.zeros(( X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
#     for imagenum in range(X_train.shape[0]):
#         X_grey_train[imagenum] = rgb2Gray(X_train[imagenum])
#     X_train = X_grey_train
#     return X_train

X_train = np.array([preprocess(image) for image in X_train], dtype=np.float32)
X_test = np.array([preprocess(image) for image in X_test], dtype=np.float32)
X_valid = np.array([preprocess(image) for image in X_valid], dtype=np.float32)

# Normalization: Scaling pixels to be from 0.1 to 0.9
# X_train = (X_train*(0.9-0.1)/255.) + 0.1
# X_test = (X_test*(0.9-0.1)/255.) + 0.1
# X_valid = (X_valid*(0.9-0.1)/255.) + 0.1




# Suffle:
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train,y_train)

X_valid, y_valid = shuffle(X_valid,y_valid)
X_test, y_test = shuffle(X_valid,y_valid)




# Model Architecture

import tensorflow as tf
from tensorflow.contrib.layers import flatten
EPOCHS = 10
BATCH_SIZE = 128


def LeNet(x,keepprob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu,stddev=sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16),mean=mu,stddev=sigma)),
    'wfc1': tf.Variable(tf.truncated_normal(shape=(5*5*16,120), mean=mu, stddev=sigma)),
    'wfc2': tf.Variable(tf.truncated_normal(shape=(120,84) , mean=mu , stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(84,43), mean=mu,stddev=sigma ))}

    biases = {
    'bc1': tf.Variable(tf.zeros((6))),
    'bc2': tf.Variable(tf.zeros((16))),
    'bfc1': tf.Variable(tf.zeros((120))),
    'bfc2': tf.Variable(tf.zeros(84)),
    'out': tf.Variable(tf.zeros((43)))}

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6. (Change in width and height comes from using VALID padding)
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc1']
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6 Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # TODO: Layer 2: Convolutional. Input = 14x14x6  Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1,weights['wc2'],strides=[1,1,1,1],padding='VALID') + biases['bc2']
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 600. Output = 120.
    fc1 = tf.add(tf.matmul(conv2,weights['wfc1']),biases['bfc1'])

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    
    fc1 = tf.nn.dropout(fc1,keepprob)

    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1,weights['wfc2']),biases['bfc2'])

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2,keepprob)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.

### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# Train, Validate, and Test the Model

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)


# Training Pipline
rate = 0.001

keep_prob = tf.placeholder(tf.float32)

logits = LeNet(x, keep_prob)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)

loss_operation = tf.reduce_mean(cross_entropy)
# The tf.train.AdamOptimizer uses Kingma and Ba's Adam algorithm to control the 
# learning rate. Adam offers several advantages over the simple 
# tf.train.GradientDescentOptimizer. Foremost is that it uses moving 
# averages of the parameters (momentum); Bengio discusses the reasons for why 
# this is beneficial in Section 3.1.1 of this paper. Simply put, this enables Adam 
# to use a larger effective step size, and the algorithm will converge to this step 
# size without fine tuning.

#The main down side of the algorithm is that Adam requires more computation to be performed for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient); and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter). A simple tf.train.GradientDescentOptimizer could equally be used in your MLP, but would require more hyperparameter tuning before it would converge as quickly.
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# Model Evalution

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, dropout):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
            
        validation_accuracy = evaluate(X_valid, y_valid, 1.)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")



# evaluate the Model

# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))

#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))



# Test a Model on New Images
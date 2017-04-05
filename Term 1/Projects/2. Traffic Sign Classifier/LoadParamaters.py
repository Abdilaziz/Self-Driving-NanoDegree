import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
# import scipy.misc as misc
import os
from tensorflow.contrib.layers import flatten


# imagePath = ''
images = np.array([mpimg.imread('images/' + name) for name in os.listdir('images/')], dtype=np.float32)

print(images.shape)

# image = mpimg.imread('example.jpg')

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

grey_images = np.array([preprocess(image) for image in images], dtype=np.float32)

# import matplotlib.pyplot as plt

# plt.imshow(grey_images[0][:,:,0],cmap='gray')
# plt.show()

x = tf.placeholder(tf.float32, (None, 32, 32, 1))

print(grey_images.shape)



# image = rgb2Gray(image)

# print(image.shape)
# Images = np.zeros((1, image.shape[0], image.shape[1], 1 ))
# print(Images.shape)
# Images[0] = image


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

logits = LeNet(x, 1.)

# estimate = tf.argmax(tf.nn.softmax(logits))

estimate = tf.nn.softmax(logits)

# saver = tf.train.import_meta_graph('lenet.meta')

with tf.Session() as sess:
    saver = tf.train.Saver()

    estimate = tf.argmax(estimate,1)

    saver.restore(sess, tf.train.latest_checkpoint('.'))

    # Show the values of weights and bias
    print('Logits:')
    print(sess.run(estimate, feed_dict={x: grey_images}))
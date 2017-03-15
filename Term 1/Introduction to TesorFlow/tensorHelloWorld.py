import tensorflow as tf

# In tensorflow, data isn't stored as integers, floats, or strings.
# They are encapsulated by an object called a Tensor.



# Create TensorFlow object called tensor
# the tensor returned by tf.constant is called a constant tensor because
# the value of the tensor never changes.
hello_constant = tf.constant('Hello World!')

# placeholders create tensors where the value needs to be feed in when run in a session.
# a placeholder tensors can't be modified.
# returns a tensor that gets its value from data passed to tf.sessions.run()
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
w = tf.placeholder(tf.float64)

a = tf.add(5,2) # takes 2 numbers, 2 tensors or 1 of each

# tf.subtract(tf.constant(2.0),tf.constant(1)) will fail due to data type difference
# to cast data types
b = tf.subtract(tf.cast(tf.constant(2.0),tf.int32),tf.constant(1))

# the variable class creates a tensor with an initial value (neccessary) that can be modified.
# this is more like the normal python variable
c = tf.Variable(5)
# the vairable tensor stores its state in the session, so you
# need to initialize the state of the tensor manually,
# tf.global_variables_initializer()

n_features = 120
n_labels = 5
# truncated_normal is a random variable with a normal distribution, whose magnitude is no more than 2 standard deviations away from the mean
weights = tf.Variable(tf.truncated_normal(n_features,n_labels))
# Intializes to a tensor with an array of all zeroes.
bias = tf.Variable(tf.zeroes(n_labels))

# initializes all the variables from the graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    # The Session is an enviornment for running a graph
    # The session is in charge of allocating the operations
    # to GPUs and CPUs including remote machines


    sess.run(init)
    # Placeholder gives you a non-constant tensor
    # You use feed_dict because over time you will want
    # your tensorflow model to take in different datasets
    # with differnet parameters
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})



    # This line evaluates the tensor in a session
    # it creates a sesssion instance using tf.Session
    # sess.run evalues the tensor and returns the result
    #output = sess.run(hello_constant)
    print(output)
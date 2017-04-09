# Keras Testing

# Sequential Model
# Sequential is a class wrapper for the neural network model.
# has the functions fit(), evaluate(), compile()
from keras.models import Sequential

model = Sequential()

# A keras layer is just like a neural network layer
# there are Fully Connected Layers, max pool layers, ad activation layers
# add layers to the model using the model's add() function


#1st Layer - Add a flatten layer
# input model is 32x32x3 so flattened output is 3072
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd Layer - Add a fully connected layer
# output is 100
model.add(Dense(100))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

#4th Layer - Add a fully connected layer
model.add(Dense(60))

#5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))
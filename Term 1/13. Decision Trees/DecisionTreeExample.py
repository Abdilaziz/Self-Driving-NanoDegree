from sklearn import tree

X = [[0,0],[1,1]]
Y = [0,1]


# some parameters
# min_samples_split
# Controls how much examples need to be a part of that tree node to keep splitting
# which is why it can lead to complex decision boundaries
# Default is 2

#Entropy
# Controls how a Decision Tree decides where to split the data
# Defintion: Measure of impurity in a bunch of examples
# Example Speed Limit

# criterion
# used to calculate information gain
# default is gini, but it can handle entropy as well.

clf = tree.DecisionTreeClassifier(min_samples_split=2)

clf.fit(X,Y)

pred = clf.predict([[2.,2.]])


# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred,labels_test)



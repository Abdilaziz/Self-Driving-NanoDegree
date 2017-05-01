from sklearn import svm
X= [[0,0],[1,1]]
y=[0,1]
# getting classifier

# parameters for SVC() are important when getting a decision surface that 
# is non linear
# kernel, C and gamma
# C parameter controls the tradeoff between smooth decision boundary and
# classifying training points correctly
# A large C value will get more training points correct



clf = svm.SVC()
clf.fit(X,y)



print(clf.predict([[2.,2.]]))

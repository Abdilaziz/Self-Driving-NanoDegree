#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn import svm

clf = svm.SVC(kernel="rbf",C=10000.0)
#making the training set smaller increases training time
# These lines effectively slice the training dataset down to 1% of 
# its original size, tossing out 99% of the training data.
# speed is useful in real-time applications
# features_train = features_train[:int(len(features_train)/100)] 
# labels_train = labels_train[:int(len(labels_train)/100)] 
t0 = time()
clf.fit(features_train,labels_train)
print("prediction time:",round(time()-t0,3),"s")

pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

howMuchAre1=0
for predic in pred:
	if(predic==1):
		howMuchAre1=howMuchAre1+1



print(howMuchAre1)
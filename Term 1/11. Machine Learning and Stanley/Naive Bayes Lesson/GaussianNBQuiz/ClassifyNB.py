from sklearn.naive_bayes import GaussianNB
def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    # clf.predict(features_test) Test prediction with test data, Test data shouldnt be included in features_train to give accurate results
   # print("Accuracy is: " + clf.score(features_test, labels_test)) Gives accuracy with test data and its labels

    return clf
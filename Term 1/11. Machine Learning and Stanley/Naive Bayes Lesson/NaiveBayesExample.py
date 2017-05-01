

#Naive Baise Examle
# import numpy as np
# # Array of points on Scatter Plot (Features)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# #Array of Labels, good or bad
# Y = np.array([1, 1, 1, 2, 2, 2])
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB() #Called a Classifier
# clf.fit(X, Y) # Fits Features and Labels (Always there for supervised classification problems)

# print(clf.predict([[-0.8, -1]])) # Tries to predict what label this data point belongs to

# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y)) #Tries to Fit The 

# print(clf_pf.predict([[-0.8, -1]]))












# import numpy as np

# a = np.arange(15).reshape(3, 5)

# print(a)

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
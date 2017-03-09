
# We have 3 features to judge wheter the student should get accepted or rejected
# to a graduate program, their gre score, gpa, and rank of the undergraduate school

# the rank of the undergraduate school is currently just 1 to 4, where 1 is the highest,
# however, we know that rank is not some weighted value, but is categorical

# We can clean up the data to change the structure to have the rank classified.
# We will add a column for each rank where a 1 in that column corresponds to an undergraduate
# school of that rank

# Encoding the data into 1's and 0's is important for our neural network

# We also need to properly scale the gpa and gre data so that the values have a zero mean, and
# a standard deviation of one. (In order for the sigmoid function to be valueable, values need to be between
# -1 and 1 with a mean of zero)

# The gradient of really small and large inputs is zero, which means that the gradient 
# descent step will go to zero too. Since the GRE and GPA values are fairly large, 
# we have to be really careful about how we initialize the weights or the gradient 
# descent steps will die off and the network won't train. Instead, if we standardize 
# the data, we can initialize the weights easily and everyone is happy.

# Now that the data is ready, we see that there are six input features: 
# gre, gpa, and the four rank dummy variables.


import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']
# Grid Search Brute Force Algorithm for best fit model

# load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
from pandas import *

# Load data
iris = datasets.load_iris()
features = iris.data
print("Features", features)
target = iris.target
print("Target" , target)

# Create logistic regression
logistic = linear_model.LogisticRegression()

# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']

# Create a range of candidate penalty hyperparameter values
C = np.logspace(0, 4, 10)

# Create a dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Fit grid search
best_model = gridsearch.fit(features, target)

# View best hyperparameters
print('Best penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Predict target vector
best_model.predict(features)

# Filename: 01-two-spiral-mlp.py
# Author: George Edwards c3167656
# Date: 2017-04-11
# Course: COMP3330
#
# Description: This file creates an ANN (sklearn MLPClassifier object) to classify the two spiral dataset.
# It reads in the data from a CSV file to be used to train an MLPCLassifier.
# matplotlib is then used to plot all points on a finite x,y space
# through the classifier to visualise its predictions (how it divides/categorises the space; should be a spiral)

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from random import shuffle


# reads CSV file
def from_csv(filename, newline='', delimiter=','):
    x = []
    y = []

    # "with" closes the file when it's done
    with open(filename, newline=newline) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            x.append(list(map(float, row[:-1])))
            y.append(int(row[-1]))

    return x, y

# shuffle the data (permute)
# First it combines the inputs and the categories (x and y)
# shuffles, then returns them separated: zip(*combined) - the '*' character unzips 'combined'
def shuffle_data(input, tar):
    combined = list(zip(input, tar))
    shuffle(combined)
    return zip(*combined)

# Create points over an xy space
# which will be fed intot he classifier and coloured
# according to the classifiers categorisation on each xy point
def activate_over_xy():
    x_range = np.arange(-7, 7, 0.1)
    y_range = np.arange(-7, 7, 0.1)
    x, y = np.meshgrid(x_range, y_range)
    xy = np.array([x.flatten(), y.flatten()]).T
    predictions = myClassifyer.predict(xy)
    colours = {0: 'y', 1: 'b'}
    # colours = {0: 'm', 1: 'c'}
    z = list(map(colours.get, predictions))
    plt.scatter(x, y, c=z)
    return plt


x, y = from_csv("spiralsdataset.csv")
# x, y = from_csv("spiralsdataset-2.csv")   # a more dense data set we generated ourselves
x, y = shuffle_data(x, y)


# sklearn's MLPClassifier's parameters.
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
nn = MLPClassifier(
                   # activation='relu',       # Activation function for the hidden layer.
                   solver='lbfgs',           # The solver for weight optimization.
                   max_iter=50000,          # Maximum number of iterations. The solver iterates until convergence (determined
                                            # by ‘tol’) or this number of iterations.
                   hidden_layer_sizes=(150,150),  # network configuration
                   # #tol=0.0000001,        # Tolerance for the optimization. When the loss or score is not improving by at least
                                            # tol for two consecutive iterations, unless learning_rate is set to ‘adaptive’,
                                            # convergence is considered to be reached and training stops.
                   verbose=True,     # Whether to stdout training progress messages.
                   # warm_start=False,  # begin not from RNG'ing the weights but from the previous iteration of training
                   alpha=0.4,      # L2 penalty (regularization term) parameter.
                   # random_state=0     # RNG seed used for initial weights
                   )

# Grid Search: set up parms to iterate over
# parameters_grid = { 'hidden_layer_sizes': [
#                                             #[20,20,20,20,5],
#                                             [32,32,32,32,32],
#                                             #[40,40,40,40,5]
#                                           ]
#                    # 'alpha': [0.01,0.001,0.0001]
#                   }

# Grid search
# nnGridSearch = GridSearchCV(nn, param_grid=parameters_grid)

# train ANN
# nnGridSearch.fit(xTrain, yTrain)
nn.fit(x, y)
# nn.fit(xTrain, yTrain)

# print reports
# print('best parms: ', nn.best_params_)
print(classification_report(y, nn.predict(x)))

# now plot results
# myClassifyer = nnGridSearch
myClassifyer = nn
plt = activate_over_xy()
plt.show()
# Filename: 01-two-spiral-svm.py
# Author: George Edwards c3167656
# Date: 2017-04-11
# Course: COMP3330
#
# Description: This file creates a SVM (sklearn svm object) to classify the two spiral dataset.
# It reads in the data from a CSV file, uses it to train the SVM, then matplotlib plots
# all points on a finite x,y space by running them through the classifier, colouring how it categorises the point,
# thus, visualising the classifier's categorisation (hopefully a spiral)

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from random import shuffle
import pickle


TEST_TO_TRAIN_RATIO = 0.3


# shuffle and zip/tar? the input data
def shuffle_data(input, tar):
    combined = list(zip(input, tar))
    shuffle(combined)
    return zip(*combined)


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

# uses myClassifier to predict points in xy space
def activate_over_xy():
    x_range = np.arange(-7, 7, 0.1)
    y_range = np.arange(-7, 7, 0.1)
    x, y = np.meshgrid(x_range, y_range)
    xy = np.array([x.flatten(), y.flatten()]).T
    predictions = myClassifier.predict(xy)
    colours = {0: 'y', 1: 'b', 2:'r'}
    z = list(map(colours.get, predictions))
    plt.scatter(x, y, c=z)
    return plt


x, y = from_csv("spiralsdataset.csv")
# x, y = from_csv("spiralsdataset-2.csv") # more dense data we generated ourselves.
# x, y = from_csv("3-Spiral.csv") # more dense data we generated ourselves.

x, y = shuffle_data(x, y)

# split data up
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=TEST_TO_TRAIN_RATIO)

# using default sklearn SVM parameters
mySvm = svm.SVC(    #kernel='rbf',
                    C=25
                    )

# Grid Search: set up parms to iterate over
parameters_grid = {
                    #'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    #'C': [5,10,20,25,30,35,40]
                  }

# Grid search
# mySvm = GridSearchCV(mySvm, param_grid=parameters_grid)


# train SVM
mySvm.fit(xTrain, yTrain)
# mySvm.fit(x, y)
myClassifier = mySvm # myClassifier will be used for plotting later

# print reports
# print('best parms: ', mySvm.best_params_)
# print('TRAINING DATA', classification_report(yTrain, myClassifier.predict(xTrain)))
# print('TESTING DATA', classification_report(yTest, myClassifier.predict(xTest)))
print('TESTING DATA', classification_report(yTest, myClassifier.predict(xTest)))
# print('TESTING DATA', classification_report(y, myClassifier.predict(x)))

# now plot results
plt = activate_over_xy()
plt.show()

#dump the classifier
# pickle.dump(mySvm, open('01-twospiral-svm.dat', 'wb'))
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from random import shuffle


TEST_TO_TRAIN_RATIO = 0.7


# shuffles two separate lists; will be used to shuffle the input data where list1:=values & list2:=categories
def shuffle_data(list1, list2):
    combined = list(zip(list1, list2))
    shuffle(combined)
    return zip(*combined)   # '*' unzips the list


# reads CSV file
def from_csv(filename, newline='\n', delimiter=';'):
    dataValues = []
    dataCategories = []

    # "with" closes the file when it's done
    with open(filename, newline=newline) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        # next(csv_reader, None)  # skip header row
        for row in csv_reader:
            dataValues.append(list(row[:-1]))
            dataCategories.append(row[-1])  # don't include the last value (aka: the last 'column') -they're categories



    return dataValues, dataCategories


# using pandas to read csv (might end p beign useless)
# newlist = pd.read_csv("q2-dataset.csv")
# print(newlist)

# Read in data from csv
# dataValues, dataCategories = from_csv("q2-dataset.csv")
dataValues, dataCategories = from_csv("q2-dataset-2.csv")   # modified dataset
dataValues, dataCategories = shuffle_data(dataValues, dataCategories)

# print(dataValues)
# print(dataCategories)

# split data up
dataTrainValues, dataTestValues, dataTrainCategories, dataTestCategories = train_test_split(dataValues, dataCategories, test_size=TEST_TO_TRAIN_RATIO)

# using default sklearn SVM parameters
mySvm = svm.SVC(
                    # kernel='rbf',
                    C=1,
                    verbose=True
                )

# Grid Search: set up parms to iterate over
parameters_grid = {
                    'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                    'C': [1,2,3,4,5,6,7,8,9,10]
                  }

# Grid search
# mySvm = GridSearchCV(mySvm, param_grid=parameters_grid)

# train SVM
mySvm.fit(dataTrainValues, dataTrainCategories)

# print reports
# print('best parms: ', mySvm.best_params_)
print('TRAINING DATA', classification_report(dataTrainCategories, mySvm.predict(dataTrainValues)))
print('TESTING DATA', classification_report(dataTestCategories, mySvm.predict(dataTestValues)))
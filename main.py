import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def readDataFrame():
    return pd.read_csv('IMDB Dataset.csv')


def split(dataFrame):
    x, y = dataFrame.iloc[:, :-1], dataFrame.iloc[:, [-1]]  # split feature and label
    X_train, X_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, test_size=0.3, stratify=y)
    X_validate, X_test, y_validate, y_test = train_test_split(X_rem, y_rem, train_size=1/3, test_size=2/3,stratify=y_rem)
    return X_train, X_validate, X_test, y_train, y_validate, y_test


dataFrame = readDataFrame()
X_train, X_test, X_validate, y_train, y_validate, y_test = split(dataFrame)
print(y_train.value_counts())
print(y_test.value_counts())
print(y_validate.value_counts())

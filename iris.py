import pandas as pd
import numpy as np
import pickle
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# read data
iris = pd.read_csv('Iris.csv')
iris = iris.drop('Id', axis=1)

X = iris.drop('Species', axis=1)
y = iris['Species']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))

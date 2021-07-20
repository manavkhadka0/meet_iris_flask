import pandas as pd
import numpy as np
import pickle
import graphviz
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# read data
iris = pd.read_csv('Iris.csv')
iris = iris.drop('Id', axis=1)
#Data prepare
X = iris.drop('Species', axis=1)
y = iris['Species']
#labelencoder on output
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# Splitting into traning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model making
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

graph = graphviz.Source(tree.export_graphviz(clf, out_file=None,
                                             feature_names=iris.columns[:4],
                                             class_names=iris.columns[4],
                                             filled=True, rounded=True,
                                             special_characters=True))
bytes = graph.pipe(format='png')
with open('./static/images/tree.png','wb') as file:
    file.write(bytes)




# Performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
con_mat = confusion_matrix(y_test,y_pred)

class_matrix= classification_report(y_test,y_pred,output_dict=True)

set_matrix = class_matrix['0']
ver_matrix = class_matrix['1']
vir_matrix = class_matrix['2']

set_matrix ={key:round(value,2) for key,value in set_matrix.items()}
ver_matrix ={key:round(value,2) for key,value in ver_matrix.items()}
vir_matrix ={key:round(value,2) for key,value in vir_matrix.items()}


pickle.dump(clf, open('model.pkl', 'wb'))

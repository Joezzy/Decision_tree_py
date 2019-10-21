import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score

clf=tree.DecisionTreeClassifier()

X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf = clf.fit(X_train, Y_train)

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

Y_prediction = clf.predict(X_test)

print("Prediction for Decision Tree: ",Y_prediction)
print("Accuracy:",accuracy_score(Y_test,Y_prediction))

X_new = np.array([[190,70,43]])
prediction=clf.predict(X_new)
print (prediction)
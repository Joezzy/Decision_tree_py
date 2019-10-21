import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from IPython.display import Image
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris=load_iris()
X=iris.data
y=iris.target

#print(iris.feature_names)
clf=tree.DecisionTreeClassifier(random_state=0)
model=clf.fit(X,y)
from sklearn.tree import export_graphviz
#dot_data=tree.export_graphviz(clf,out_file='tree_limited.dot', feature_names=iris.feature_names, class_names=iris.target_names)

export_graphviz(clf, out_file='tree_limited.dot', feature_names = iris.feature_names,class_names = iris.target_names,\
                rounded = True, proportion = False, precision = 2, filled = True)

from subprocess import check_call
check_call(['dot','-Tpng','tree_limited.dot','-o','OutputFile.png'])


print('end')


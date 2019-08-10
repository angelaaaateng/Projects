

'''Import Libraries'''

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pprint import pprint
from yellowbrick.features import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix
import sklearn.model_selection as model_selection
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot
from random import sample
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced
from sklearn import tree
import pydotplus
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn import inspection
import mlxtend

from mlxtend.evaluate import feature_importance_permutation
print(__doc__)
print(__name__)

if __name__ == "__main__":
    '''
    Navigate Directory
    '''
    print("* Navigating through directory")
    os.chdir('/Users/angelateng/Documents/GitHub/Projects/Covertype_Prediction/Data')
    print(os.getcwd())
    print(__name__)
    print('Directory navigated')
    input = open("./covtype.data")

from preprocess_data_old import preprocess

#preprocess()

def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X_train, y_train)
    dtree = DecisionTreeClassifier( random_state=42)
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    print("Decision Tree Train Accuracy:", metrics.accuracy_score(y_train, dtree.predict(X_train)))
    print("Decision Tree Test Accuracy:", metrics.accuracy_score(y_test, dtree.predict(X_test)))
    y_pred = dtree.predict(X_test)
    print('* Decision Tree Classification Report')
    print(classification_report(y_test, y_pred))
    print('* Decision Tree Confusion Matrix')
    print(confusion_matrix(y_test,predictions))
    dtree_train_accuracy = dtree.predict(X_train)
    dtree_test_accuracy = dtree.predict(X_test)
    return(dtree_train_accuracy, dtree_test_accuracy)

def random_forest(df_normalized_w_target):


    X = df_normalized_w_target[list(df_normalized_w_target.columns)[7:-1]]
    print(X.shape)
    Y=df_normalized_w_target[list(df_normalized_w_target.columns)[-1]]
    print(Y.shape)

    perm_feat_imp = X.iloc[:,[0,5,9,3,12,13,4,23,7,10]]

    X, y = make_imbalance(perm_feat_imp, Y,
                      sampling_strategy={1: 2700, 2: 2700, 3: 2700, 4:2700, 5:2700, 6:2700, 7:2700},
                      random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc = rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print(rfc_pred.shape)
    y_pred =  rfc.predict(X_test)
    print(y_pred.shape)

    rf_train_acc = metrics.accuracy_score(y_train, rfc.predict(X_train))
    rf_test_acc = metrics.accuracy_score(y_train, rfc.predict(X_test))
    print ("Random Forest Train Accuracy:", metrics.accuracy_score(y_train, rfc.predict(X_train)))
    print ("Random Forest Test Accuracy:", metrics.accuracy_score(y_test, rfc.predict(X_test)))
    print(confusion_matrix(y_test,rfc_pred))
    print(classification_report(y_test,rfc_pred))
    return(rf_train_acc, rf_test_acc)

def predict():
    X_train, X_test, y_train, y_test, df_normalized_w_target = preprocess()
    dtree_train_accuracy, dtree_test_accuracy = decision_tree(X_train, X_test, y_train, y_test)
    rf_train_acc, rf_test_acc = random_forest(df_normalized_w_target)
    print('* Prediction Complete')
    return(dtree_train_accuracy, dtree_test_accuracy,rf_train_acc, rf_test_acc )

predict()

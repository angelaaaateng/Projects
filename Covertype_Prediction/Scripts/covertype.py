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
import mca
from random import sample
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn import tree
import pydotplus
print(__doc__)

'''Navigate Directory'''
import os
os.chdir('/Users/angelateng/Documents/GitHub/Projects/Covertype_Prediction/Data')
os.getcwd()
print('Directory navigated')
data = open("./covtype.data")


def read_data(data):
    '''
    Read Data
    '''
    data = pd.read_csv("covtype.data", header=None)
    return(data);
print('* Data loaded');

#data = pd.read_csv("covtype.data", header=None)
#print(data["Cover_Type"])


def create_dummies(data):
    '''
    Create Dummy Variables and Normalize
    '''
    cov_dummy = pd.get_dummies(data['Cover_Type'])
    df4 = pd.concat([cov_dummy, data], axis = 1)
    df4_column_names = list(df4.columns)
    df4_column_names.remove('Cover_Type')
    # Normalize all columns
    x = df4.loc[:, df4.columns != 'Cover_Type'].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(data=x_scaled, columns=df4_column_names)
    return(df_normalized);
print('* Dataframe normalized');


def norm_target(df_normalized):
    '''
    Add Target Variable to Normalized Data
    '''
    df_normalized_w_target = pd.concat([df_normalized, df4['Cover_Type']], axis=1)
    df_dummy = df_normalized_w_target
    df_dummy = df_dummy.drop(['Cover_Type'], axis=1)
    return(df_dummy);
print('* Normalized target variable added to dummy df')


def xy_labels(df_normalized_w_target):
    '''
    Get X and Y labels
    '''
    X=df_normalized_w_target[list(df_normalized_w_target.columns)[7:-1]]
    Y=df_normalized_w_target[list(df_normalized_w_target.columns)[-1]]
    return(X,Y)
print('* X,Y defined')

RANDOM_STATE = 42


def train_test(X,Y):
    '''
    Split into training and testing sets
    '''
    X, y = make_imbalance(X, Y,
                      sampling_strategy={1: 2700, 2: 2700, 3: 2700, 4:2700, 5:2700, 6:2700, 7:2700},
                      random_state=RANDOM_STATE)
    print('Training target statistics: {}'.format(Counter(y_train)))
    print('Testing target statistics: {}'.format(Counter(y_test)))
    return(X_train, X_test, y_train, y_test)
print('* Dataset split')


def preprocessing(data):
    '''
    Data Pre-processing
    '''
    data = read_data(data);
    df_normalized = create_dummies(data);
    df_dummy = norm_target(df_normalized);
    X, Y = xy_labels(df_normalized_w_target);
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test(X,Y);
    return(X_train, X_test, y_train, y_test)
print('* Complete')
#print('Training target statistics: {}'.format(Counter(y_train)))
#print('Testing target statistics: {}'.format(Counter(y_test)))


def dtree(X_train, X_test, y_train, y_test):
    '''
    Decision Tree
    '''
    clf = DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X_train, y_train)
    dtree = DecisionTreeClassifier( random_state=42)
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    print ("Decision Tree Train Accuracy:", metrics.accuracy_score(y_train, dtree.predict(X_train)))
    print ("Decision Tree Test Accuracy:", metrics.accuracy_score(y_test, dtree.predict(X_test)))
    y_pred = dtree.predict(X_test)
    print('Accuracy of decision tree classifier on test set: {:.2f}'.format(dtree.score(X_test, y_test)))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test,predictions))
    return(y_pred)
print('* Decision Tree Processed')

#print(preprocessing(data))


def print_tree(X_train, X_test, y_train, y_test, df_normalized_w_target):
    '''
    Visualize Decision Tree
    '''
    feature_list = list(df_normalized_w_target.columns)[7:-1]
    features = list(feature_list)
    dtree_baseline = DecisionTreeClassifier(max_depth=3, random_state=42)
    dtree_baseline.fit(X_train,y_train)
    dot_data = StringIO()
    export_graphviz(dtree_baseline, out_file=dot_data,feature_names=features,filled=True,rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return(Image(graph[0].create_png()))
print('* Decision Tree Printed')


def rf_baseline(X_train, X_test, y_train, y_test):
    '''
    Random Forest
    '''
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    y_pred =  rfc.predict(X_test)
    print ("Random Forest Train Accuracy Baseline:", metrics.accuracy_score(y_train, rfc.predict(X_train)))
    print ("Random Forest Test Accuracy Baseline:", metrics.accuracy_score(y_test, rfc.predict(X_test)))
    print(confusion_matrix(y_test,rfc_pred))
    print(classification_report(y_test,rfc_pred))
    return(y_pred)


def rf_feat(rfc):
    '''
    Feature selection for RF
    '''
    from sklearn import inspection
    import mlxtend
    from mlxtend.evaluate import feature_importance_permutation
    importance_vals = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
    indices = np.argsort(importance_vals)[::-1]
    ranked_index = [feature_list[i] for i in indices]
    plt.figure(figsize=(12,8))
    plt.title("Random Forest feature importance")
    plt.bar(range(X.shape[1]), importance_vals[indices],
            yerr=std[indices], align="center")
    #plt.xticks(range(X.shape[1]), indices)
    plt.xticks(range(X.shape[1]), (ranked_index), rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.ylim([0, 0.5])
    return(plt.show())

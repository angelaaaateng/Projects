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

'''Read Data'''
def read_data(data):
    data = pd.read_csv("covtype.data", header=None)
    return(data);
print('Data loaded');

'''Create Dummy Variables and Normalize'''
def create_dummies(data):
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
print('Dataframe normalized');

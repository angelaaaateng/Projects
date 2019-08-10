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
print(__doc__)


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



def read_data():
    '''
    Read Data
    '''

    data = pd.read_csv("./covtype.data", header=None)
    # set column names
    cols = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
       'vertical_distance_to_hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4',
       'Soil_Type_1',
        'Soil_Type_2',
        'Soil_Type_3',
        'Soil_Type_4',
        'Soil_Type_5',
        'Soil_Type_6',
        'Soil_Type_7',
        'Soil_Type_8',
        'Soil_Type_9',
        'Soil_Type_10',
        'Soil_Type_11',
        'Soil_Type_12',
        'Soil_Type_13',
        'Soil_Type_14',
        'Soil_Type_15',
        'Soil_Type_16',
        'Soil_Type_17',
        'Soil_Type_18',
        'Soil_Type_19',
        'Soil_Type_20',
        'Soil_Type_21',
        'Soil_Type_22',
        'Soil_Type_23',
        'Soil_Type_24',
        'Soil_Type_25',
        'Soil_Type_26',
        'Soil_Type_27',
        'Soil_Type_28',
        'Soil_Type_29',
        'Soil_Type_30',
        'Soil_Type_31',
        'Soil_Type_32',
        'Soil_Type_33',
        'Soil_Type_34',
        'Soil_Type_35',
        'Soil_Type_36',
        'Soil_Type_37',
        'Soil_Type_38',
        'Soil_Type_39',
        'Soil_Type_40',
       'Cover_Type']
    data.columns = cols
    #print(data['Cover_Type'])
    print('* Data loaded')
    cov_dummy = pd.get_dummies(data['Cover_Type'])
    df4 = pd.concat([cov_dummy, data], axis = 1)
    df4_column_names = list(df4.columns)
    df4_column_names.remove('Cover_Type')
    #pprint(data.columns)
    #print(df4)
    #print(read_data)
    #print(df4_column_names)
    return(data, df4, df4_column_names);

#read_data()

def normalize_data(df4, df4_column_names):
    x = df4.loc[:, df4.columns != 'Cover_Type'].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(data=x_scaled, columns=df4_column_names)
    #print(df_normalized)
    df_normalized_w_target = pd.concat([df_normalized, df4['Cover_Type']], axis=1)
    df_dummy = df_normalized_w_target
    df_dummy = df_dummy.drop(['Cover_Type'], axis=1)
    print('* Data Normalized')
    return(df_normalized, df_normalized_w_target)

def sample_data(df_normalized_w_target):
    X=df_normalized_w_target[list(df_normalized_w_target.columns)[7:-1]]
    Y=df_normalized_w_target[list(df_normalized_w_target.columns)[-1]]
    X, y = make_imbalance(X, Y,
                      sampling_strategy={1: 2700, 2: 2700, 3: 2700, 4:2700, 5:2700, 6:2700, 7:2700},
                      random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    #print(X_train, X_test, y_train, y_test
    print('* Data Sampled')
    return(X_train, X_test, y_train, y_test)

def preprocess():
    data, df4, df4_column_names = read_data()
    df_normalized, df_normalized_w_target = normalize_data(df4, df4_column_names)
    #print(normalized_data)
    X_train, X_test, y_train, y_test = sample_data(df_normalized_w_target)
    print('* Data Preprocessing Complete')
preprocess()

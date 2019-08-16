import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.datasets import make_imbalance


# import pickle

from joblib import dump, load
#from data_preprocessing import preprocess

# from data_preprocessing import preprocess


# data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test, y_test = preprocess("./covtype.data")
#
# def initialize_sample(df_normalized_w_target, X_test, y_test):
#     X, y = make_imbalance(X_test, y_test,
#                       sampling_strategy={1: 2700, 2: 2700, 3: 2700, 4:2700, 5:2700, 6:2700, 7:2700},
#                       random_state=42)
#     X_train, X_test_new, y_train, y_test_new = train_test_split(X_test, y_test, random_state=42)
#     #print(X_train, X_test, y_train, y_test
#     print('* Data Sampled')
#     print(X_train, X_test_new, y_train, y_test_new)
#     return(X_train, X_test_new, y_train, y_test_new)


def hyper_param_rf(X_test, y_test):
    rfc = RandomForestClassifier(n_estimators=300, max_depth=20,
     min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
    rf_model2 = load('./grid_search_optimal.joblib')
    print('* Joblib model loaded -- picklejar')
    # rfc.fit(X_test, y_test)
    # n_optimal_param_grid = {
    # 'bootstrap': [True],
    # 'max_depth': [20], #setting this so as not to create a tree that's too big
    # #'max_features': [2, 3, 4, 10],
    # 'min_samples_leaf': [1],
    # 'min_samples_split': [2],
    # 'n_estimators': [300]
    # }
    rfc.fit(X_test, y_test)
    load('./grid_search_optimal.joblib')
    # grid_search_optimal = GridSearchCV(estimator = rfc, param_grid = n_optimal_param_grid,
                          # cv = 3, n_jobs = -1, verbose = 2)
    # grid_search_optimal.fit(X_test, y_test)
    # grid_search_optimal.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    # print ("Random Forest Train Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_train, grid_search_optimal.predict(X_train)))
    print ("Random Forest Test Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_test, rfc.predict(X_test)))
    conf_mat = confusion_matrix(y_test,y_pred)
    class_rept = classification_report(y_test,y_pred )
    print(confusion_matrix(y_test,y_pred ))
    print(classification_report(y_test,y_pred ))
    # rfc_train_acc = metrics.accuracy_score(y_train, rfc.predict(X_train))
    rfc_test_acc = metrics.accuracy_score(y_test, rfc.predict(X_test))

    rf_model = dump(rfc, './grid_search_optimal.joblib')
    # saved_model = pickle.dumps(grid_search_optimal)

    # filename = 'model.pkl'
    # pickle.dump(model, open(filename, 'wb'))
    # print(rfc_train_acc, rfc_test_acc, y_pred)
    print("* File pickled using joblib -- picklejar")
    return(rfc_test_acc, y_pred, class_rept, conf_mat)

if __name__ == "__main__":
    '''
    Navigate Directory
    '''
    #print("* Navigating through directory")
    #os.chdir('/Users/angelateng/Documents/GitHub/Projects/Covertype_Prediction/Data')
    #print(os.getcwd())
    #print(__name__)
    print('Directory navigated')
    #input = open("./covtype.data")

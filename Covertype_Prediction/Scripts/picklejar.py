import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import accuracy_scores

# import pickle

#from data_preprocessing import preprocess



def hyper_param_rf(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    n_optimal_param_grid = {
    'bootstrap': [True],
    'max_depth': [20], #setting this so as not to create a tree that's too big
    #'max_features': [2, 3, 4, 10],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [300]
    }
    grid_search_optimal = GridSearchCV(estimator = rfc, param_grid = n_optimal_param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
    grid_search_optimal.fit(X_train, y_train)
    y_pred = grid_search_optimal.predict(X_test)
    print ("Random Forest Train Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_train, grid_search_optimal.predict(X_train)))
    print ("Random Forest Test Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_test, grid_search_optimal.predict(X_test)))
    print(confusion_matrix(y_test,y_pred ))
    print(classification_report(y_test,y_pred ))
    rfc_train_acc = metrics.accuracy_score(y_train, rfc.predict(X_train))
    rfc_test_acc = metrics.accuracy_score(y_test, rfc.predict(X_test))

    # filename = 'model.pkl'
    # pickle.dump(model, open(filename, 'wb'))
    return(rfc_train_acc, rfc_test_acc, y_pred)

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

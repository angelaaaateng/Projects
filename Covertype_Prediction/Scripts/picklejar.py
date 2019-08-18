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
import matplotlib.pyplot as plt

from joblib import dump, load

def hyper_param_rf_pickle(X_test, y_test, model):
    # rfc = RandomForestClassifier(n_estimators=300, max_depth=20,
    #  min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
    # rf_model2 = load('./grid_search_optimal.joblib')
    rfc = model
    print('* Joblib model loaded -- picklejar')

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
    print("* Saving results in an image in picklejar...")
    fig = plt.figure()
    plt.matshow(conf_mat)
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('./confusion_matrix.jpg')

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

from flask import Flask, make_response, request, render_template
import pandas as pd
import io
import csv

from data_preprocessing import read_data, normalize_data, preprocess

from picklejar import hyper_param_rf_pickle
from predict import hyper_param_rf_predict
from build_model import sample_data
from joblib import dump, load

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report,confusion_matrix
import os

import jinja2
import matplotlib

matplotlib.use("TKAgg")
print(matplotlib.get_backend())

import matplotlib.pyplot as plt

app = Flask(__name__)

#model = None

# def load_model():
#     '''
#     Load the pre-trained word2vec model and define it as a global variable
#     that we can use after startup
#     '''
#     global model
    # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    # print("* Model loaded successfully")

@app.route('/')
def form():
    return("""
        <html>
            <body>
                <h1><center>Random Forest Multi-Class Classifier</h1>

                <form action="/datafeedback" method="post" enctype="multipart/form-data">
                    <center> <input type="file" name="data_file" />
                    <br>
                    <p> </p>
                    <center> <input type="submit"/>
                </form>
            </body>
        </html>
    """)


def load_model():
    '''
    Load the model and define it as a global variable
    that we can use after startup
    '''
    global model
    model = load('./grid_search_optimal.joblib')
    # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("* Model loaded successfully")


@app.route('/datafeedback', methods=["GET", "POST"])
def transform_view():

    # START COMMENT BLOCK
    print("* Requesting data -- API")
    f = request.files['data_file']
    # print(f)
    if not f:
        return("No file selected. Please choose a CSV file and try again.")

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    print("* Processing csv_input -- API")
    # print(stream)
    # print(csv_input)
    df = pd.DataFrame(csv_input, index=None, columns=None)
    # df = pd.DataFrame([csv_input], index=None, columns=None)
    # print(df.head())

    data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test_new, y_test_new = preprocess(df)
    print('* Data Preprocessing Complete Flask -- API')

    # model = load('./grid_search_optimal.joblib')
    print('* Joblib model loaded -- API')



    # X_train, X_test, y_train, y_test = sample_data(df_normalized_w_target)
    # print('* Data Sampled')
    # X_train, X_test_new, y_train, y_test_new = initialize_sample(df_normalized_w_target, X_test, y_test)
    # print("* Data Initialized for First Pickle")

    rfc_test_acc, y_pred, class_rept, conf_mat = hyper_param_rf_pickle(X_test_new, y_test_new, model)
    print("* Hyperparameter search complete -- API")
    conf_mat = confusion_matrix(y_test_new,y_pred)
    class_rept = classification_report(y_test_new,y_pred )
    print("* Conf Mat and Class Rept Defined -- API")
    # END COMMENT BLOCK

    # print(df4.head())
    # print(df4_column_names)
    # return(data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test, y_test)

    # print('* API: Data Preprocessing Completed')

    # class_rept = "hello i am class_rept"
    # conf_mat = "hello i am conf_mat"
    #return confusion matrix
    # return("""
    #     <html>
    #         <body>
    #             <h1> Model Performance Results </h1>
    #             <h2><center>Classification Report</h2>
    #             <form action="/datafeedback" method="post" enctype="application/x-www-form-urlencoded">
    #             <p>{{class_rept}}</p>
    #             <h2><center>Confusion Matrix</h2>
    #             <p>{conf_mat}</p>
    #         </body>
    #     </html>
    # """)
    print("* Saving results in an image....")
    # fig = plt.figure()
    # plt.matshow(conf_mat)
    # plt.title('Confusion Matrix')
    # plt.show()
    # plt.savefig('./confusion_matrix.jpg')

    return('* CSV File Submitted -- Running API')

    #return(f)
    # return(rfc_test_acc, y_pred, class_rept, conf_mat)
    # template_dir = '/Users/angelateng/Documents/GitHub/Projects/Covertype_Prediction/Scripts'
    # loader = jinja2.FileSystemLoader(template_dir)
    # environment = jinja2.Environment(loader=loader)

    # return(render_template('page.html', conf_mat=conf_mat, class_rept=class_rept))


if __name__ == '__main__':
    app.debug = True
    load_model()
    app.run()

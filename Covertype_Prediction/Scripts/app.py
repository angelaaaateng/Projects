from flask import Flask, make_response, request, render_template
import pandas as pd
import io
import csv

from data_preprocessing import read_data, normalize_data, preprocess

from picklejar import initialize_sample

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

                <form action="/datafeecback" method="post" enctype="multipart/form-data">
                    <center> <input type="file" name="data_file" />
                    <br>
                    <p> </p>
                    <center> <input type="submit"/>
                </form>
            </body>
        </html>
    """)

# @app.route('/datafeecback', methods=["POST"])
@app.route('/datafeecback', methods=["GET", "POST"])
def transform_view():
    # if request.method == "POST":
    #     df = pd.read_csv(request.files.get('data_file'))
    print("* Requesting data")
    f = request.files['data_file']
    # print(f)
    if not f:
        return("No file selected. Please choose a CSV file and try again.")
    # stream = io.BytesIO(f.stream.read().decode("UTF8"), newline=None)
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    print("* Processing csv_input")
    # print(stream)
    # print(csv_input)
    df = pd.DataFrame(csv_input, index=None, columns=None)
    # df = pd.DataFrame([csv_input], index=None, columns=None)
    # print(df.head())

    data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test, y_test = preprocess(df)

    #preprocess(f)
    #def preprocess(csv_file):
    # data, df4, df4_column_names = read_data(df)
    # data, df4, df4_column_names = read_data(stream)
    # df_normalized, df_normalized_w_target, X_test, y_test = normalize_data(df4, df4_column_names)
    print('* Data Preprocessing Complete Flask')
    X_train, X_test_new, y_train, y_test_new = initialize_sample(df_normalized_w_target, X_test, y_test)
    print("* Data Initialized for First Pickle")
    # print(df4.head())
    # print(df4_column_names)
    # return(data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test, y_test)

    # print('* API: Data Preprocessing Completed')

    #return confusion matrix
    return('* CSV File Submitted -- Running')
    #return(f)

# def preprocess(input):
    # print('* CSV File Submitted -- Running')
    # print([row for row in input])


if __name__ == '__main__':
    app.debug = True
    app.run()

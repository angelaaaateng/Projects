from flask import Flask, make_response, request
import io
import csv

from data_preprocessing import read_data, normalize_data, preprocess

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
                    <center> <input type="submit" />
                </form>
            </body>
        </html>
    """)

@app.route('/datafeecback', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    #return(f)
    if not f:
        return("No file selected. Please choose a CSV file and try again.")

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)

    #preprocess(f)
    #def preprocess(csv_file):
    data, df4, df4_column_names = read_data(f)
    df_normalized, df_normalized_w_target = normalize_data(df4, df4_column_names)
    print('* Data Preprocessing Complete Flask')
    return(data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test, y_test)

    print('* API: Data Preprocessing Completed')

    #return confusion matrix
    return('* CSV File Submitted -- Running')
    #return(f)

# def preprocess(input):
#     print('* CSV File Submitted -- Running')
#     print([row for row in input])


if __name__ == '__main__':
    app.debug = True
    app.run()

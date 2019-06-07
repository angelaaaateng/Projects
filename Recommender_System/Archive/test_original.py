import gensim
from flask import Flask
import io
import flask
import json

app = Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("* Model loaded successfully")

def find_vocab():
    return model.vocab

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.data:
            # read the image in PIL format
            word = json.loads(flask.request.data)["word"].encode('ascii','ignore')

            data["included"] = word in find_vocab()

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading gensim model and Flask starting server..."
        "please wait until server has fully started"))
    app.debug = True
    load_model()
    app.run()

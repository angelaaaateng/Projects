import gensim
from flask import Flask, request
import io
import flask
import json

import Pangea_RS_Cleaned
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
model = None

def load_model():
    # load the pre-trained Word2Vec model
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
    print("* Model loaded successfully")

def find_vocab():
    return model.vocab

@app.route("/pangea", methods=["POST"])
def predict(self, user_post):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure a user post was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.data:
            # read the json user post in the correct format
            #word = json.loads(flask.request.data)["word"].encode('ascii','ignore')
            with open('firstPost.json') as fresh_data:
                user_post = json.load(flask.request.fresh_data)

            data["included"] = user_post in find_vocab()

            # indicate that the request was a success
            data["success"] = True


    return flask.jsonify(user_post)

#input is new user post and pickle
#class Pangea_RS_Cleaned(Resource):
    #def get(self, user_post, second_number):
        #return {'Recommendations': Pangea_RS_Cleaned.generate_recommendations(user_post)}
         # return the data dictionary as a JSON response
    #return flask.jsonify(data)
#return depends on what pangea wants
#api.add_resource(Pangea_RS_Cleaned, '/Pangea_RS_Cleaned/<user_post>')
#api.add_resource(Pangea_RS_Cleaned, '/Pangea_RS_Cleaned/')

if __name__ == "__main__":
    print(("* Loading gensim model and Flask starting server..."
        "please wait until server has fully started"))
    app.debug = True
    load_model()
    app.run()

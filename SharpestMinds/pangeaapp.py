import gensim
from flask import Flask
import io
import flask
import json

import Pangea_RS_Cleaned
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("* Model loaded successfully")

#input is new user post and pickle
class Pangea_RS_Cleaned(Resource):
    def get(self, user_post, second_number):
        return {'Recommendations': Pangea_RS_Cleaned.generate_recommendations(user_post)}
#return depends on what pangea wants
#api.add_resource(Pangea_RS_Cleaned, '/Pangea_RS_Cleaned/<user_post>')
api.add_resource(Pangea_RS_Cleaned, '/Pangea_RS_Cleaned/')

if __name__ == '__main__':
     app.run()

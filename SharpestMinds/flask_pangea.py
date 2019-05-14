'''
Recommender System - Flask App for Pangea.App
'''

'''
Importing modules and initializing flask
'''

import gensim
from flask import Flask
import io
import flask
import json
from Pangea_Final_Script import generate_recommendations

'''
Import Flask class and create an instance of this class
'''
app = Flask(__name__)
#for more information on flask easy startup see http://flask.pocoo.org/docs/1.0/quickstart/
model = None

def load_model():
    '''
    load the pre-trained word2vec model and define it as a global variable
    that we can use after startup

    input: none ; but maximum size of the model
    '''
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("* Model loaded successfully")

def find_vocab():
    '''
    Returns a model vocuabulary (dict)
    '''
    return model.vocab
    print("* Model Vocab Loaded")
    #return(None)

'''
Specify the app route using a route decorater to
bind the predict function to a url.
'''
@app.route("/pangeaapp", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned
    # view
    data = {"success": False}
    print("* Initialization ok")

    # ensure that a json request was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.data:
            # read json title
            title = json.loads(flask.request.data)["title"]
            #encode('ascii','ignore')
            print("*Input Title: ")
            print(title)
            data["recommendations"] = generate_recommendations(title, model)
            #print(data["recommendations"])

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

'''
'''

if __name__ == "__main__":
    print(("* Loading gensim model and Flask starting server..."
        "please wait until server has fully started"))
    app.debug = True
    #set app debug to true so that whenever a change is made on .py code,
    #it reflects on server/client terminal tools
    load_model()
    app.run()

'''
Notes for running flask app:
- remember to check both the client and server side
- use the curl command to call on the flask app:
curl localhost:5000/pangeaapp -d '{"title": "Teach me how to cook!"}' -H 'Content-Type: application/json'
'''

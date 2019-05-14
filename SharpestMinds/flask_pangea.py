import gensim
from flask import Flask
import io
import flask
import json
from Pangea_Final_Script import generate_recommendations

app = Flask(__name__)
model = None

def load_model():
    # load the pre-trained word2vec model
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("* Model loaded successfully")

def find_vocab():
    return model.vocab
    print("* Model Vocab Loaded")
    #return(None)

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

if __name__ == "__main__":
    print(("* Loading gensim model and Flask starting server..."
        "please wait until server has fully started"))
    app.debug = True
    load_model()
    app.run()

#check client and server side
# need this command
#curl -X POST -H "Content-Type: application/json" --data '{"word": "eggfd" }' http://localhost:5000/pangeaapp

# curl localhost:5000/pangeaapp -d '{"title": "bar"}' -H 'Content-Type: application/json'
#curl localhost:5000/pangeaapp -d '{"title": "Teach me how to cook!"}' -H 'Content-Type: application/json'

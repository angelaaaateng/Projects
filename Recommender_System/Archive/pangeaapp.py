import Pangea_RS_Cleaned

from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

#input is new user post and pickle
class Pangea_RS_Cleaned(Resource):
    def get(self, user_post):
        return {'Recommendations': Pangea_RS_Cleaned.generate_recommendations(user_post)}
#return depends on what pangea wants
#api.add_resource(Pangea_RS_Cleaned, '/Pangea_RS_Cleaned/<user_post>')
api.add_resource(Pangea_RS_Cleaned, '/Pangea_RS_Cleaned/')

if __name__ == '__main__':
     app.run()

# Pangea Recommender System


## Recommender System - Marketplace Matching Script for Pangea.App


- Recommender_lib.py - Python Script containing Word2Vec model and recommender system
- app.py - flask application for deploying the recommender system; Flask App for Pangea.App
- requirements.txt - requirements needed to run the model


## Notes for running flask app:
- remember to check both the client and server side
- use the curl command to call on the flask app:
curl localhost:5000/pangeaapp -d '{"title": "Teach me how to cook!"}' -H 'Content-Type: application/json'

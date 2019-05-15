#!/usr/bin/env python
# coding: utf-8
#generate requirements: pip freeze > requirements.txt

'''
Recommender System - Marketplace Matching Script for Pangea.App
'''


'''
Importing Libraries
'''
import numpy as np
import pandas as pd
from pandas import DataFrame

import sys, os

import json
from pandas.io.json import json_normalize

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv

import gensim
from gensim import corpora
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from collections import defaultdict
from pprint import pprint
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import gensim.downloader as api

#these packages mostly used for visualization submitted to team on ipynb
import re
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter, attrgetter, add
from pprint import pprint

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from adjustText import adjust_text
import string

import pickle
import operator
import csv


#we load the model in the flask app, but can load it here if stand alone
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)


def vectorize_and_store_existing_titles(model):
'''
Vectorize and store existing titles in legacy Pangea database

Input: Word2Vec Model (.bin)
Output: Vectorized Titles (.pkl)

Note: can split this up into two different functions later
'''
    raw = pd.read_csv("allPostData.csv", header=0);
    titles = raw['title'];
    post_titles = [title for title in titles];
    post_titles = set(post_titles);
    tokens = [[word for word in title.lower().split()] for title in post_titles];
    clean_words = [[word.translate(str.maketrans('', '', string.punctuation)) for word in title] for title in tokens]
    stoplist = set(stopwords.words('english'));
    titles_nostopwords = [[word for word in title if word not in stoplist] for title in clean_words];
    filtered_word_list = [[word for word in title if word in model.vocab] for title in titles_nostopwords];
    dictionary = dict(zip(post_titles, filtered_word_list))
    vectorized_titles = pd.DataFrame(columns=["Titles", "Vectors"])
    for title in post_titles:
        word_vecs = [model[word] for word in dictionary[title]]
        if len(word_vecs) == 0:
            title_vec = [np.zeros(300)]
        else:
            title_vec = normalize(sum(word_vecs).reshape(1, -1))
        vectorized_titles = vectorized_titles.append({'Titles': title, 'Vectors': title_vec}, ignore_index=True)
    #note that we are saving the df with the original raw (not cleaned) titles
    vectorized_titles.to_pickle("/Users/angelateng/Google_Drive/SharpestMinds_dropbox/SharpestMinds/vectorized_titles.pkl")
    return(vectorized_titles)


def vectorize_new_title(title, model):
'''
Vectorize each new title as a user/student/company creates a new post

Input:
- title from user query curl command (str)
- model (.bin)

Output: json_vectorized_title_df (dict)

'''
    #uncomment this if using this script as stand alone
    #ranked_titles_load = pd.read_csv("./ranked_titles.csv")
    #json_df = pd.DataFrame.from_dict(json_normalize(title), orient='columns')
    #title = json_df["title"][0]

    json_tokens = [word for word in title.lower().split()]
    json_clean_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in json_tokens]
    stoplist = set(stopwords.words('english'));
    json_titles_nostopwords = [word for word in json_clean_words if word not in stoplist]
    json_preprocessed = [word for word in json_titles_nostopwords if word in model.vocab]
    json_title_vectors = {}
    json_vectorized_title_df = pd.DataFrame(columns=["Titles", "Vectors"])
    json_word_vecs = [model[word] for word in json_preprocessed]
    #manually normalizing the word vectors here since normalize command here didn't work
    if len(json_preprocessed) == 0:
            json_title_vec = [np.zeros(300)]
    else:
        json_title_vec = normalize(sum(json_word_vecs).reshape(1, -1))
    json_vectorized_title_df = json_vectorized_title_df.append({'Titles': title, 'Vectors': json_title_vec}, ignore_index = True)
    if not os.path.isfile('/Users/angelateng/Google_Drive/SharpestMinds_dropbox/SharpestMinds/ranked_titles.csv'):
        json_vectorized_title_df.to_csv (r'/Users/angelateng/Google_Drive/SharpestMinds_dropbox/SharpestMinds/ranked_titles.csv', index = None, header=True)
    else:
        json_vectorized_title_df.to_csv (r'/Users/angelateng/Google_Drive/SharpestMinds_dropbox/SharpestMinds/ranked_titles.csv', mode='a', index = None, header=False)
    return(json_vectorized_title_df)


def rank_existing_titles(vectorized_title):
'''
Load the current titles in the Pangea database,
and then rank them by similarity to the latest user query

Input: vectorized titles (.pkl)
Output: sorted title vectors (dict)
'''
    ranked_titles = {}
    other_titles = pd.read_pickle("./vectorized_titles.pkl")
    for index,row in other_titles.iterrows():
        ranked_titles[row['Titles']] = sum(row['Vectors'][0]*vectorized_title['Vectors'][0][0])
        # did the dot product using sum() and * because np.dot was behaving weirdly for some reason.
    sorted_title_vecs = sorted(ranked_titles.items(), key=operator.itemgetter(1), reverse=True)
    return(sorted_title_vecs)


def generate_recommendations(title, model):
'''
Final function call API that puts together the
prior 3 functions in a neat mega-function

Input:
- User inputted titles via curl command (str)
- Google News Vectors Model (.bin)

Output:
- ranked titles (dict)
note that this will print in the terminal on the client side
'''
    #error checking here
    #data = request.get_json(force=True)
    #convert json to df
    #with open('firstPost.json') as fresh_data:
    #    user_post = json.load(fresh_data)
    #title = user_post["title"]
    vectorized_title = vectorize_new_title(title, model)
    ranked_titles = rank_existing_titles(vectorized_title)
    other_titles = pd.read_pickle("./vectorized_titles.pkl")
    other_titles.append({"Titles": title, "Vectors": vectorized_title}, ignore_index=True)

    with open("./ranked_titles.csv", "w", newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for title in ranked_titles:
            wr.writerow([ranked_titles, title])


    #print(ranked_titles)
    #ranked_titles = pd.DataFrame(rank_existing_titles(vectorized_title), columns = ["Title", "Similarity Score"])

    return(ranked_titles)
    print("*COMPLETE")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:54:18 2021

@author: ravishrawal
"""

# ---- IMPORTS -----

#get stemmer
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import re

#get english stopwords
from nltk.corpus import stopwords
    
stop_words_list = set(stopwords.words('english'))

#get dictionary
import enchant
d = enchant.Dict('en_US')

#get tfidf vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()


#get pandas
import pandas as pd

#get sklearn models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
    
    
# ---- HELPERS ----

def write_pickle(path_in, file_name, var_in):
    import pickle
    pickle.dump(var_in, open(path_in + file_name, "wb"))
    
def open_pickle(path_in, file_name):
    import pickle
    tmp = pickle.load(open(path_in + file_name, "rb"))
    return tmp
    
def csv_to_df(f_url):
    import pandas as pd
    reviews_df = pd.read_csv(f_url)
    
    #remove irrelevant columns
    del reviews_df["Unnamed: 0"]
    del reviews_df["link"]
    del reviews_df["helpful-count"]
    # del reviews_df["dates"]
    
    #rename columns with hyphens to underscores
    reviews_df.columns = [i.replace('-', '_') for i in reviews_df.columns]
    
    #remove nan values
    nan_indexes = reviews_df[ pd.isna(reviews_df['summary'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['pros'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['cons'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['advice_to_mgmt'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['overall_ratings'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['company'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    empty_indexes = reviews_df[ reviews_df['summary']=="" ].index
    reviews_df.drop(empty_indexes, inplace = True)
    
    empty_indexes = reviews_df[ reviews_df['pros']=="" ].index
    reviews_df.drop(empty_indexes, inplace = True)
    
    empty_indexes = reviews_df[ reviews_df['cons']=="" ].index
    reviews_df.drop(empty_indexes, inplace = True)
    
    empty_indexes = reviews_df[ reviews_df['advice_to_mgmt']=="" ].index
    reviews_df.drop(empty_indexes, inplace = True)
    
    empty_indexes = reviews_df[ reviews_df['overall_ratings']=="" ].index
    reviews_df.drop(empty_indexes, inplace = True)
    
    empty_indexes = reviews_df[ reviews_df['company']=="" ].index
    reviews_df.drop(empty_indexes, inplace = True)
    
    return reviews_df

def to_lowercase(inp_str):
    #func to lower case a str
    return inp_str.lower()

def rem_sp_char(inp_str):
    # rem special characters except hyphenated words or a letter followed by +
    inp_str = re.sub('[^a-zA-Za-zA-Z-a-zA-z+]+', ' ',inp_str)
    return inp_str
    
def rem_sw(inp_str):
    #func to remove all stopwords

    #remove any matching words
    str_rem_sw = [word for word in inp_str.split() if word not in stop_words_list]

    #reconstruct string & return
    return ' '.join(str_rem_sw)

def rem_non_eng(inp_str):
    #func to ensure words are in english dictionary
    
    #check if word is valid
    str_rem_nonwords = [word for word in inp_str.split() if d.check(word)]

    #reconstruct string & return
    return ' '.join(str_rem_nonwords)

def stem_str(inp_str):
    #func to stem words
    
    #stem words
    stemmed_words = [ps.stem(word) for word in inp_str.split()]
    
    #reconstruct string & return
    return ' '.join(stemmed_words)

def vec_count(column, inp_df, base_path):
    #func to convert column to count vectorization
    column_count = inp_df[column]
    count_vec = pd.DataFrame(count_vectorizer.fit_transform(column_count).toarray())
    count_vec.columns = count_vectorizer.get_feature_names()
    
    #write output
    write_pickle(base_path, column+"_countvec.pkl", count_vec)
    
    return count_vec

def vec_tfidf(column, inp_df, base_path):
    #func to convert column to tfidf vectorization
    column_tfidf = inp_df[column]
    tf_idf_vec = pd.DataFrame(tfidf_vectorizer.fit_transform(column_tfidf).toarray())
    tf_idf_vec.columns = tfidf_vectorizer.get_feature_names()
    
    #write output
    write_pickle(base_path, column+"_tfidf.pkl", tf_idf_vec)
    
    return tf_idf_vec

def split_data(x, y, split_ratio = 0.2):
    print('splitting data... \n')
    return train_test_split(x, y, test_size=split_ratio, random_state=0)

def train_model(x_train, y_train, model='nb'):
    print('training model... \n')
    if model == 'nb':
        model_trained = MultinomialNB().fit(x_train, y_train)
    elif model == 'rf':
        model_trained = RandomForestClassifier(max_depth=10, class_weight='balanced').fit(x_train, y_train)
    
    return model_trained
    
def predict(model, x_test, y_test):
    print('predicting using NB...')
    pred = model.predict(x_test)
    print('Results: \n', metrics.classification_report(y_test, pred), '\n')
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 10:42:57 2021

@author: ravishrawal
"""

#0. load data
from utils import *

base_path = "/Users/ravishrawal/Desktop/Columbia MS/Summer B/QMSS NLP/final_project/"
reviews_df = open_pickle(base_path, "reviews_data_cleaned_empty_removed.pkl")


#1. vectorize

#1a. count vectorize
#vectorize summary (use stemmed to reduce size)
summary_count = vec_count("summary_stemmed", reviews_df, base_path)
# summary_tfidf = open_pickle(base_path, "summary_stemmed_tfidf.pkl")


#1b. tf-idf
# summary_tfidf = vec_tfidf("summary_stemmed", reviews_df, base_path)
summary_tfidf = open_pickle(base_path, "summary_stemmed_tfidf.pkl")


# #2. split train-test
x = summary_count
y = reviews_df['overall_ratings']

x_train, x_test, y_train, y_test = split_data(x, y, split_ratio=0.2)


#3. train
model_trained = train_model(x_train, y_train, model = 'rf')
# write_pickle(base_path, "NB_trained.pkl", NB_trained)


#4. test
model_pred = predict(model_trained, x_test, y_test)


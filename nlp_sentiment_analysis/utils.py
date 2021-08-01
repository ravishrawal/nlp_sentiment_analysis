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
try:
    import enchant
    d = enchant.Dict('en_US')
except:
    # print('enchat is not installed')
    pass
    
    
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
    del reviews_df["dates"]
    
    #rename columns with hyphens to underscores
    reviews_df.columns = [i.replace('-', '_') for i in reviews_df.columns]
    
    #remove nan values
    nan_indexes = reviews_df[ pd.isna(reviews_df['summary'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['pros'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['cons'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['advice-to-mgmt'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['overall-ratings'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
    nan_indexes = reviews_df[ pd.isna(reviews_df['company'] ) ].index
    reviews_df.drop(nan_indexes, inplace = True)
    
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

def filter_none(df):
    # func to find none in these ratings, delete the whole row
    # if none exists in this ratings

    # return the filterd dataframe

    # print(set(df["overall_ratings"])) #{1.0, 2.0, 3.0, 4.0, 5.0}
    # print(set(df["work_balance_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
    # print(set(df["culture_values_stars"])) #{'3.0', 'none', '2.0', '4.0', '5.0', '1.0'}
    # print(set(df["carrer_opportunities_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
    # print(set(df["comp_benefit_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
    # print(set(df["senior_mangemnet_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}

    mask = df["overall_ratings"] == 'none'
    mask |= df["work_balance_stars"] == 'none'
    mask |= df["culture_values_stars"] == 'none'
    mask |= df["carrer_opportunities_stars"] == 'none'
    mask |= df["comp_benefit_stars"] == 'none'
    mask |= df["senior_mangemnet_stars"] == 'none'
    df = df[ ~mask ]

    return df

def max_length(df):

    comments = ["summary_stemmed", "pros_stemmed", "cons_stemmed"]

    max_len = 0
    for comment in comments:
        for element in df[comments]:
            tmp_len = len(element)
            max_len = tmp_len if tmp_len > max_len else max_len
    return max_len
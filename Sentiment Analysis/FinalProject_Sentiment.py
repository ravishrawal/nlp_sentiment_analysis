#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 18:54:34 2021

@author: Ruiqi Li
"""

from utils import *

base_path = "/Users/aprillee/Desktop/G5067 Natural Language Processing/Final Project/" 

#Apply the function to get sentiment scores for summary, pros, and cons
#review_data = open_pickle(base_path, "reviews_data_cleaned.pkl")
#
#review_data["sum_stem_sent"] = review_data.summary_stemmed.apply(score_sent)
#
#review_data["pros_stem_sent"] = review_data.pros_stemmed.apply(score_sent)
#
#review_data["cons_stem_sent"] = review_data.cons_stemmed.apply(score_sent)

#review_data["advice_stem_sent"] = review_data.advice_to_mgmt_stemmed.apply(score_sent)

#write_pickle(base_path, "review_with_sent_score.pkl", review_data)


senti = open_pickle(base_path, "review_with_sent_score.pkl")
with_date = open_pickle(base_path, "reviews_data_cleaned_w_dates.pkl")


import pandas as pd
from datetime import datetime
with_date['year'] = pd.to_datetime(with_date['dates'], errors='coerce').dt.year

data_with_year = pd.concat([senti, with_date['year']], axis=1)
#write_pickle(base_path, "review_w_sent_and_year.pkl", data_with_year)



#Rating over time for each company

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,9))
data_with_year.groupby(['year', 'company'])['overall_ratings'].mean().unstack().plot(ax=ax)
_ = plt.xlabel("Years")
_ = plt.ylabel("Overall Ratings")
_ = plt.title("Overall Ratings per Year")

fig, ax = plt.subplots(figsize=(15,9))
data_with_year.groupby(['year', 'company'])['sum_stem_sent'].mean().unstack().plot(ax=ax)
_ = plt.xlabel("Years")
_ = plt.ylabel("Summary Sentiment Scores")
_ = plt.title("Summary Sentiment Scores per Year")

fig, ax = plt.subplots(figsize=(15,9))
data_with_year.groupby(['year', 'company'])['pros_stem_sent'].mean().unstack().plot(ax=ax)
_ = plt.xlabel("Years")
_ = plt.ylabel("Pros Comments Sentiment Scores")
_ = plt.title("Pros Comments Sentiment Scores per Year")


fig, ax = plt.subplots(figsize=(15,9))
data_with_year.groupby(['year', 'company'])['cons_stem_sent'].mean().unstack().plot(ax=ax)
_ = plt.xlabel("Years")
_ = plt.ylabel("Cons Comments Sentiment Scores")
_ = plt.title("Cons Comments Sentiment Scores per Year")
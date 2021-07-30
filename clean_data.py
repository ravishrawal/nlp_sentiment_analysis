#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:14:06 2021

@author: ravishrawal
"""

from utils import *

base_path = "/Users/ravishrawal/Desktop/Columbia MS/Summer B/QMSS NLP/final_project/"

test = 'the work-life balance is great! Love teh culture here. a+, 5 stars'

# convert to df & remove irrelevant columns

# reviews_df = csv_to_df(base_path + "employee_reviews.txt")

# clean text & tokenize

# -- summary column --
# reviews_df["summary"] = reviews_df.summary.apply(to_lowercase)
# reviews_df["summary"] = reviews_df.summary.apply(rem_sp_char)
# reviews_df["summary"] = reviews_df.summary.apply(rem_non_eng)
# reviews_df["summary"] = reviews_df.summary.apply(rem_sw)
# reviews_df["summary_stemmed"] = reviews_df.summary.apply(stem_str)

# -- pros column --
# reviews_df["pros"] = reviews_df.pros.apply(to_lowercase)
# reviews_df["pros"] = reviews_df.pros.apply(rem_sp_char)
# reviews_df["pros"] = reviews_df.pros.apply(rem_non_eng)
# reviews_df["pros"] = reviews_df.pros.apply(rem_sw)
# reviews_df["pros_stemmed"] = reviews_df.pros.apply(stem_str)

# -- cons column --
# reviews_df["cons"] = reviews_df.cons.apply(to_lowercase)
# reviews_df["cons"] = reviews_df.cons.apply(rem_sp_char)
# reviews_df["cons"] = reviews_df.cons.apply(rem_non_eng)
# reviews_df["cons"] = reviews_df.cons.apply(rem_sw)
# reviews_df["cons_stemmed"] = reviews_df.cons.apply(stem_str)

# -- advice_to_mgmt column --
# reviews_df["advice_to_mgmt"] = reviews_df.advice_to_mgmt.apply(to_lowercase)
# reviews_df["advice_to_mgmt"] = reviews_df.advice_to_mgmt.apply(rem_sp_char)
# reviews_df["advice_to_mgmt"] = reviews_df.advice_to_mgmt.apply(rem_non_eng)
# reviews_df["advice_to_mgmt"] = reviews_df.advice_to_mgmt.apply(rem_sw)
# reviews_df["advice_to_mgmt_stemmed"] = reviews_df.advice_to_mgmt.apply(stem_str)

# save output
write_pickle(base_path, "reviews_data_cleaned.pkl", reviews_df)
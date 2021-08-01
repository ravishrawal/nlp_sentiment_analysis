#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:31:06 2021

@author: zhiyuan ma
"""
import pdb
from utils import *
from transformers import BertTokenizer
from bert_dataset import CustomDataset, CustomDataset_fast
from bert_model import BERTClass
from bert_train import train_model
from torch.utils.data import  DataLoader
import torch
import numpy as np

# We will use 3 columns of the data frame, that is
# "summary", "pros", "cons", and "advice_to_mgmt" (actualy their tokens)
# to predict 6 ratings, that is
# 'overall_ratings', 'work_balance_stars', 'culture_values_stars',
# 'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars'
comments = ["summary_stemmed", "pros_stemmed", "cons_stemmed"]
ratings = ["overall_ratings", "work_balance_stars", "culture_values_stars",
            "carrer_opportunities_stars", "comp_benefit_stars", "senior_mangemnet_stars"]


# specify device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# define hyper-parameters
MAX_LEN = 15 # it is the result of max_length(df) 
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 1e-5

# load data frame
base_path = './'
train_dataset = open_pickle(base_path, "bert_train.pkl")
val_dataset = open_pickle(base_path, "bert_val.pkl")


training_set = CustomDataset_fast(
    dataframe = train_dataset, titles = comments, targets = ratings, device = device
)
validation_set = CustomDataset_fast(
    dataframe = val_dataset, titles = comments, targets = ratings, device = device
)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)

# initialize all modules
model = BERTClass()
model.to(device)
for p in model.bert.parameters(): # free the bert to speed up training
    p.require_grads = False
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE)


# train
checkpoint_path = './models/current_checkpoint.pt'
best_model = './models/best_model.pt'
trained_model = train_model(
    1, EPOCHS, np.Inf, training_loader, validation_loader, 
    model, optimizer,checkpoint_path,best_model, device)


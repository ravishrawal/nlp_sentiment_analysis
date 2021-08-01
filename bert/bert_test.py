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
from bert_train import train_model, set_input
from torch.utils.data import  DataLoader
import torch
import numpy as np
from bert_train import load_ckp
from tqdm import tqdm
from sklearn.metrics import classification_report

comments = ["summary_stemmed", "pros_stemmed", "cons_stemmed"]
ratings = ["overall_ratings", "work_balance_stars", "culture_values_stars",
            "carrer_opportunities_stars", "comp_benefit_stars", "senior_mangemnet_stars"]

# specify device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# # load data frame
# base_path = './'
# test_dataset = open_pickle(base_path, "bert_val.pkl")

# test_set = CustomDataset_fast(dataframe = test_dataset, titles = comments, targets = ratings, device = device )
# test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0 }

# test_loader = DataLoader(test_dataset, **test_params)
base_path = './'
val_dataset = open_pickle(base_path, "bert_val.pkl")


validation_set = CustomDataset_fast(
    dataframe = val_dataset, titles = comments, targets = ratings, device = device
)


test_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 0
                }
validation_loader = DataLoader(validation_set, **test_params)
test_loader = validation_loader

# load model weights
best_model = './models/best_model.pt'
model = BERTClass()
checkpoint = torch.load(best_model)
model.load_state_dict(checkpoint['state_dict'])

def pred2cat(index, item):
    if item in ["overall_ratings", "culture_values_stars"]:
        return '%.1f' % (index)
    else: 
        return '%.1f' % ((index + 1) / 2)
        


def test_model(test_loader, model, device):
    preds = {item: [] for item in ratings}
    gts = {item: [] for item in ratings}
    print('############# Test Start   #############')
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader, 0)):

            data = set_input(data, device)
            output = model(data)

            for item in ratings:
                _, pred = torch.max(output[item],dim = -1)
                gt = data['targets'][item]
                preds[item].append(
                    pred2cat(
                        int(pred), item))
                gts[item].append(
                    pred2cat(
                        int(gt), item))

    for item in ratings:
        print('--------{} evaluation---------'.format(item))
        print(classification_report(gts[item], preds[item],
        # target_names= set(gts[item])
        ))
                # if gt == pred: right[item] += 1
    #         total += 1
    # right = {k: v / total for k,v in right.items()}
    # print('The prediction acc are', right)


test_model(test_loader, model.to(device), device)


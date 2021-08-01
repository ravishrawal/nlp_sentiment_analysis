#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:31:06 2021

@author: zhiyuan ma
"""
import torch, transformers
from torch.nn import Dropout, Linear, Sequential, ReLU
import pdb
# print(set(df["overall_ratings"])) #{1.0, 2.0, 3.0, 4.0, 5.0}
# print(set(df["work_balance_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
# print(set(df["culture_values_stars"])) #{'3.0', 'none', '2.0', '4.0', '5.0', '1.0'}
# print(set(df["carrer_opportunities_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
# print(set(df["comp_benefit_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
# print(set(df["senior_mangemnet_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}

# data.keys() = dict_keys(['summary_stemmed', 'pros_stemmed', 'cons_stemmed', 'targets'])
# data['targets'].keys() = dict_keys(['overall_ratings', 'work_balance_stars', 'culture_values_stars', 'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars'])
# data['summary_stemmed'].keys() = dict_keys(['ids', 'mask', 'token_type_ids'])
# ...
class BERTClass(torch.nn.Module):
    def __init__(self, ):
        super(BERTClass, self).__init__()
        self.comments = [
            'summary_stemmed', 'pros_stemmed', 'cons_stemmed']
        self.targets = [
            'overall_ratings', 'work_balance_stars', 'culture_values_stars', 
            'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars']
        
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = Dropout(0.3)

        proj_ls = [
            Sequential(
                Linear(768, 16), 
                ReLU()
                ) 
        ] * len(self.comments)
        self.proj_ls = Sequential(*proj_ls)

        self.cls = Linear(16 * len(self.comments), 50)


    def forward(self, data):
        output_ls = []
        for idx, comment in enumerate(self.comments):
            sub_data = data[comment]
            ids, mask, token_type_ids =  sub_data['ids'], sub_data['mask'], sub_data['token_type_ids']
            output = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)[1]
            output = self.dropout(output)
            output = self.proj_ls[idx](output)
            output_ls.append(output)

        output = torch.cat(output_ls, -1)
        output = self.cls(output)

        output_dict = {
            'overall_ratings': output[:,0:5],
            'work_balance_stars': output[:,5:15],
            'culture_values_stars': output[:,15:20],
            'carrer_opportunities_stars': output[:,20:30],
            'comp_benefit_stars': output[:,30:40],
            'senior_mangemnet_stars': output[:,40:50],
            }


        return output_dict

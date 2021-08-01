from bert_dataset import CustomDataset
from torch.utils.data import  DataLoader
import pandas as pd
from tqdm import tqdm
from utils import *
import torch
from transformers import BertTokenizer
import pdb

# specify device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# define hyper-parameters
MAX_LEN = 15 # it is the result of max_length(df) 

# load data frame
base_path = './'
df = open_pickle(base_path, "review_with_sent_score.pkl")

comments = ["summary_stemmed", "pros_stemmed", "cons_stemmed"]
ratings = ["overall_ratings", "work_balance_stars", "culture_values_stars",
            "carrer_opportunities_stars", "comp_benefit_stars", "senior_mangemnet_stars"]

df = filter_none(df)

# initialize dataset
train_size = 0.8
train_df = df.sample(frac = train_size, random_state=200)
val_df = df.drop(train_df.index).reset_index(drop=True)
train_df= train_df.reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

training_set = CustomDataset(
    dataframe = train_df, titles = comments, targets = ratings,
    tokenizer = tokenizer, max_len = MAX_LEN, device = device
)
validation_set = CustomDataset(
    dataframe = val_df, titles = comments, targets = ratings,
    tokenizer = tokenizer, max_len = MAX_LEN, device = device
)

data_ls = []
for data in tqdm(validation_set):
    data_ls.append(data)

write_pickle(
    path_in = './',
    file_name = 'bert_val.pkl',
    var_in = data_ls
)

data_ls = []
for data in tqdm(training_set):
    data_ls.append(data)

write_pickle(
    path_in = './',
    file_name = 'bert_train.pkl',
    var_in = data_ls
)



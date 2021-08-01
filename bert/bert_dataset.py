import torch
from torch.utils.data import Dataset
import pdb
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe, titles, targets, tokenizer, max_len, device):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.titles = titles
        self.targets = targets
        self.max_len = max_len
        # set vector space for each target
        # self.lab2vec = {target: 10 for target in targets if target != 'overall_ratings'}
        # self.lab2vec.update({'overall_ratings': 5})


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        data_dict = {}

        for title in self.titles:
            sub_data = row[title]
            inputs = self.tokenizer.encode_plus(sub_data, None, add_special_tokens=True, max_length=self.max_len, padding='max_length',return_token_type_ids=True, truncation=True)
        
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]

            sub_data_value = {
                'ids': ids, #torch.tensor(ids, dtype=torch.long),
                'mask': mask, #torch.tensor(mask, dtype=torch.long)),
                'token_type_ids': token_type_ids, #torch.tensor(token_type_ids, dtype=torch.long),
                # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
            data_dict[title] = sub_data_value

        target_dict = {}
        for target in self.targets:
            rating = row[target]
            target_vec, target_label = self.one_hot(rating, target)
            target_dict[target] = target_label

        data_dict['targets'] = target_dict

        return data_dict

    def one_hot(self, rating, target):
        if target in [ 'overall_ratings', "culture_values_stars"] :
            if type(rating) == str: rating = float(rating)
            index = int(rating) - 1
            # return torch.eye(5)[index], torch.tensor(index, dtype = torch.long) #.type(torch.float)
            return np.eye(5)[index], index
        else:
            if type(rating) == str: rating = float(rating)
            index = int(rating) * 2 - 1
            # return torch.eye(10)[index], torch.tensor(index, dtype = torch.long)
            return np.eye(10)[index], index

class CustomDataset_fast(Dataset):
    def __init__(self, dataframe, titles, targets, device):
        self.dataframe = dataframe
        self.titles = titles
        self.targets = targets
        self.device = device

    def __getitem__(self, index):
        row = self.dataframe[index]
        # return row
        data_dict = {}

        for title in self.titles:
            sub_data = row[title]
        
            ids = sub_data['ids']
            mask = sub_data['mask']
            token_type_ids = sub_data["token_type_ids"]

            sub_data_value = {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
            data_dict[title] = sub_data_value

        target_dict = row['targets']
        target_dict = {k: torch.tensor(v, dtype = torch.long) for k, v in target_dict.items()}
        # for target in self.targets:
        #     rating = row[target]
        #     target_vec, target_label = self.one_hot(rating, target)
        #     target_dict[target] = target_label

        data_dict['targets'] = target_dict

        return data_dict

    def __len__(self):
        return len(self.dataframe)

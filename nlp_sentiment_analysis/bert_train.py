import torch
import shutil, sys   
import pdb
from tqdm import tqdm

# print(set(df["overall_ratings"])) #{1.0, 2.0, 3.0, 4.0, 5.0}
# print(set(df["work_balance_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
# print(set(df["'culture_values_stars'"])) #{'3.0', 'none', '2.0', '4.0', '5.0', '1.0'}
# print(set(df["carrer_opportunities_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
# print(set(df["comp_benefit_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}
# print(set(df["senior_mangemnet_stars"])) #{'3.0', '1.5', '4.5', 'none', '2.5', '2.0', '4.0', '3.5', '5.0', '1.0'}

# data.keys() = dict_keys(['summary_stemmed', 'pros_stemmed', 'cons_stemmed', 'targets'])
# data['targets'].keys() = dict_keys(['overall_ratings', 'work_balance_stars', 'culture_values_stars', 'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars'])
# data['summary_stemmed'].keys() = dict_keys(['ids', 'mask', 'token_type_ids'])
# ...

criterion = torch.nn.CrossEntropyLoss()
comments = ["summary_stemmed", "pros_stemmed", "cons_stemmed"]
ratings = ["overall_ratings", "work_balance_stars", "culture_values_stars",
            "carrer_opportunities_stars", "comp_benefit_stars", "senior_mangemnet_stars"]

def loss_fn(outputs, target):
    loss = 0
    for item in ratings:
        loss += criterion(outputs[item], target[item])
    return loss

def set_input(data, device):
    data_dict = {}
    for title in comments:
        sub_data = data[title]
    
        ids = sub_data['ids']
        mask = sub_data['mask']
        token_type_ids = sub_data["token_type_ids"]
        sub_data_value = {
            'ids': ids.to(device), #torch.tensor(ids, dtype=torch.long).to(device),
            'mask': mask.to(device), #torch.tensor(mask, dtype=torch.long).to(device),
            'token_type_ids': token_type_ids.to(device) #torch.tensor(token_type_ids, dtype=torch.long).to(device),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
        data_dict[title] = sub_data_value

    target_dict = data['targets']
    target_dict = {k: v.to(device) for k, v in target_dict.items()}
    # target_dict = {k: torch.tensor(v, dtype = torch.long).to(device) for k, v in target_dict.items()}
    # for target in self.targets:
    #     rating = row[target]
    #     target_vec, target_label = self.one_hot(rating, target)
    #     target_dict[target] = target_label

    data_dict['targets'] = target_dict
    return data_dict

def train_model(start_epochs,  n_epochs, valid_loss_min_input, 
                training_loader, validation_loader, model, 
                optimizer, checkpoint_path, best_model_path, device):
   
  # initialize tracker for minimum validation loss
  valid_loss_min = valid_loss_min_input 
 
  for epoch in range(start_epochs, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in tqdm(enumerate(training_loader)):

        # data.keys() = dict_keys(['summary_stemmed', 'pros_stemmed', 'cons_stemmed', 'targets'])
        # data['targets'].keys() = dict_keys(['overall_ratings', 'work_balance_stars', 'culture_values_stars', 'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars'])
        # data['summary_stemmed'].keys() = dict_keys(['ids', 'mask', 'token_type_ids'])
        # ...
        data = set_input(data, device)
        output = model(data)
        # output.keys() = dict_keys(['overall_ratings', 'work_balance_stars', 'culture_values_stars', 'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars'])
        optimizer.zero_grad()
        loss = loss_fn(output, data['targets'])
        if batch_idx%5000==0:
           print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = int(train_loss + ((1 / (batch_idx + 1)) * (loss- train_loss)))
        #print('after loss data in training', loss.item(), train_loss)
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in tqdm(enumerate(validation_loader, 0)):

            output = model(data)

            loss = loss_fn(outputs, targets)
            valid_loss = int(valid_loss + ((1 / (batch_idx + 1)) * (loss - valid_loss)))
            # val_targets.extend(targets.cpu().detach().numpy().tolist()) 
            # val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
      
      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
        
        # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        # save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

    return model

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
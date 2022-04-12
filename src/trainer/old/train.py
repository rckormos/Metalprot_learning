# metal_binding_classifier

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#import the tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import RandomSampler, SequentialSampler
import torchvision
from torchvision import datasets, transforms
import torchmetrics as  metrics
import random
import time
import datetime

from PIL import Image
import pandas as pd
import numpy as np

##------ nets ----------------------------
from .models import alphafold

##------ dataset -------------------------
from .metal_binding_classifier import *

import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# set up paths (update your paths here)

# PROJ_PATH = '/home/ybwu/projects/Protein/testing/mbc/'
# DATA_PATH = PROJ_PATH + 'input/'
# out_dir = '/home/ybwu/projects/Protein/testing/mbc/Aug1221_AF1/'

#BATCH_SIZE = 16
#EPOCHS = 20

def train_model(foldno, train_df, valid_df, DATA_PATH, out_dir, BATCH_SIZE = 16, EPOCHS = 20):
    
    #Sample weight
    from collections import Counter
    y_train = train_df.label.values.astype(int)
    count=Counter(y_train)
    class_count=np.array([count[0],count[1]])
    weight=1./class_count
    weight=weight/sum(weight)
    print("class weight:", weight)
    #weight = [1.0, 1.0]
    
    #samples_weight = np.array([weight[t] for t in y_train])
    #samples_weight=torch.from_numpy(samples_weight)
    #sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # Create datasets and dataloaders
    train_set = image_set(train_df, DATA_PATH, mode='train')
    train_loader = DataLoader(
        train_set, 
        sampler = RandomSampler(train_set),
        batch_size = BATCH_SIZE,
        drop_last = True, 
      # shuffle = True,
        pin_memory=True,
        num_workers = BATCH_SIZE)
    
    valid_set = image_set(valid_df, DATA_PATH, mode='valid')
    valid_loader = DataLoader(
        valid_set,
        sampler = SequentialSampler(valid_set), 
        batch_size = BATCH_SIZE, 
        drop_last = False,
      # shuffle=False,
        pin_memory=True,
        num_workers = BATCH_SIZE)
    
    print("len of train_loader:", len(train_loader))
    print("len of valid_loader:", len(valid_loader))

    # Setting up GPU
    #device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build the model
    model = alphafold.AlphafoldNet()
    model= model.to(device)
    
    # cost_function
    cost_function = nn.BCELoss()  
    
    # optimizer
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad],lr=0.0001)
    
    # learning rate scheduler
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.01)
    num_epochs=EPOCHS
  
    iter_valid = 20       # every 10 batch
    iteration4train = 5   # only for 5 of train batches
    iteration_counted = 10000
  
    min_loss = 1000
  
    start_time = time.time()
    iteration = 0
    for epoch in range(num_epochs):
    # epoch start
      print('-'*20)
      print('Start training {}/{}'.format(epoch+1,num_epochs))
      print('-'*20)
      
    # Start Training model
      model.train()
  
      for x,y in train_loader:
          # Clear the grad
          optimizer.zero_grad()
      
          # Put x and y to GPU and get predictions
          x,y = x.to(device),y.to(device)
          outputs = model(x)
            
          # Store the loss
          loss = cost_function(outputs,y.type_as(outputs))
  
          # count and update gradients
          loss.backward()
          optimizer.step()
          #scheduler.step()
     
          #train performance 
          iteration += 1
          iteration_counted += 1
          if iteration % iter_valid == 0: 
              y_train = torch.tensor([])
              y_tpred = torch.tensor([])
              iteration_counted = 0
  
          if iteration_counted < iteration4train: 
              y_train = torch.cat([y_train, y.detach().cpu()])
              y_tpred = torch.cat([y_tpred, outputs.detach().cpu()])
  
          if iteration_counted == iteration4train: 
              train_auc = roc_auc_score(y_true=y_train, y_score=y_tpred)
              y_tpred = (y_tpred > 0.5) 
              train_acc = accuracy_score(y_true=y_train, y_pred=y_tpred)
  
              total_time = time.time() - start_time
              total_time_str = str(datetime.timedelta(seconds=int(total_time)))
              print('train batch number:', iteration, ', time:', total_time_str)
              print('train: loss {:.3f}, acc {:.3f}, auc {:.3f}'.format(
                     loss, train_acc, train_auc))
  
      print("valid performance")
      model.eval()
      epoch_valid_losses = []
      y_valid = torch.tensor([])
      y_vpred = torch.tensor([])
      for x,y in valid_loader:
          x,y  = x.to(device),y.to(device)
          outputs = model(x)
          loss = cost_function(outputs,y.type_as(outputs))
          epoch_valid_losses.append(loss.item())
          
          y_valid = torch.cat([y_valid, y.detach().cpu()])
          y_vpred = torch.cat([y_vpred, outputs.detach().cpu()])
     
      valid_loss = np.mean(epoch_valid_losses)
      valid_auc = roc_auc_score(y_score=y_vpred, y_true=y_valid)
      y_vpred = (y_vpred > 0.5)
      valid_acc = accuracy_score(y_true=y_valid, y_pred=y_vpred)
     
      # Print the result
      print('valid: loss {:.3f}, acc {:.3f}, valid_auc {:.3f}'.format(
                   valid_loss, valid_acc, valid_auc))
  
    # if epoch > 0 and float(valid_loss) < min_loss:
      if float(valid_loss) < min_loss:
          print('save the best model\n')
          min_loss = valid_loss
  
          filename_m = out_dir + 'best_model' + str(foldno) +'.pth'
          torch.save({
              'state_dict': model.state_dict(),
              'iteration': iteration,
              'epoch': epoch,
          }, filename_m)
  
    # epoch ends
  
    print('Finish training.')
    return

#--------------------------------------------------------------------------
# Make prediction on test data
def predict_test(model, test_loader, device):
    print('For test part.....')
    model.eval()
    predictions = torch.tensor([])
    y_test = torch.tensor([])
    for x,y in test_loader:
        x,y = x.to(device),y.to(device)
        predictions = torch.cat([predictions,model(x).detach().cpu()])
        y_test = torch.cat([y_test, y.detach().cpu()])
    predictions = predictions.numpy()
    return predictions, y_test

def run_test(test_df, DATA_PATH, out_dir, BATCH_SIZE = 16, EPOCHS = 20):
    # Setting up GPU

    #device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_set = image_set(test_df, DATA_PATH, mode='valid')
    test_loader = DataLoader(
        test_set,
        sampler = SequentialSampler(test_set), 
        batch_size = BATCH_SIZE, 
        drop_last = False,
        pin_memory=True,
        num_workers = BATCH_SIZE)
    print("len of test_loader:", len(test_loader))

    for foldno in range(10):
        # build the model
        model = alphafold.AlphafoldNet()
        model= model.to(device)
        # load the best model
        initial_checkpoint = out_dir + 'best_model' + str(foldno) +'.pth'
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(f['state_dict'], strict=True)
        start_iteration = f['iteration']
        start_epoch     = f['epoch']

        predictions, y_test = predict_test(model, test_loader, device)
        if foldno == 0:
            pred = predictions
        else:
            pred += predictions
    pred /= 10.0
    test_auc = roc_auc_score(y_score=pred, y_true=y_test)
    print(f"test_auc SCORE: {test_auc}")
    pred = np.where(pred > 0.5, 1, 0)
    print("confusion_matrix")
    print(confusion_matrix(y_test,pred))
    print("testset classification_report")
    print(classification_report(y_test, pred, digits=4))
    
    print('testing done!')
   
################################################################
'''

if __name__ == '__main__':

    #load the samples for both postivte and megative samples
    train_df = pd.read_csv(DATA_PATH + "samples_both.csv", header=None, names=['sample_id'])
    train_df['label'] = train_df['sample_id'].astype(str).str[-5:-4]
    print("train_df", train_df.head())

    #split the train_df into train, validation and test df
    train_df,test_df,_,_ = train_test_split(train_df,train_df,test_size=0.10,random_state=42)

    #10 fold training on the remaining train_df dataset
    train_df = train_df.reset_index(drop=True)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['label'])):
        print("FOLDS : ", n_fold)
        train = train_df.loc[train_idx]
        valid = train_df.loc[valid_idx]
        print('proteins for training: {},  for validation: {}'.format(
               train.shape[0], valid.shape[0]))
   
        train_model(n_fold, train, valid)
    print('training done!')

    run_test(test_df)

'''

def run(DATA_PATH, out_dir, BATCH_SIZE = 16, EPOCHS = 20):
    #load the samples for both postivte and megative samples
    train_df = pd.read_csv(DATA_PATH + "samples_both.csv", header=None, names=['sample_id'])
    train_df['label'] = train_df['sample_id'].astype(str).str[-5:-4]
    print("train_df", train_df.head())

    #split the train_df into train, validation and test df
    train_df,test_df,_,_ = train_test_split(train_df,train_df,test_size=0.10,random_state=42)

    #10 fold training on the remaining train_df dataset
    train_df = train_df.reset_index(drop=True)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['label'])):
        print("FOLDS : ", n_fold)
        train = train_df.loc[train_idx]
        valid = train_df.loc[valid_idx]
        print('proteins for training: {},  for validation: {}'.format(
               train.shape[0], valid.shape[0]))
   
        train_model(n_fold, train, valid, DATA_PATH, out_dir, BATCH_SIZE = 16, EPOCHS = 20)
    print('training done!')

    run_test(test_df, DATA_PATH, out_dir, BATCH_SIZE = 16, EPOCHS = 20)
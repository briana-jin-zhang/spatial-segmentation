import torch 
import numpy as np
from PIL import Image
from torchvision import transforms
import h5py
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import time
import pickle

class MaskTrainer:
    def __init__(self, device, net, 
                 train_set, 
                 val_set,  
                 batch_size, 
                 optimizer, 
                 scheduler=None,
                 loss = nn.MSELoss(), 
                 in_key = 0, 
                 target_key = 1,
                 mask_key = 2,
                 num_workers=0,
                 checkpoint_dir='./models/', 
                 exp_name='net'):
        self.device = device
        self.net = net.to(device)
        self.loss = loss
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                       num_workers=num_workers, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, 
                                     num_workers=num_workers, shuffle=False)
        self.in_key = in_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.optimizer = optimizer
        self.scheduler = scheduler
        logging_dir_name = exp_name + '_' + str(time.time()) + '/'
        self.checkpoint_dir = checkpoint_dir + logging_dir_name
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        os.mkdir(self.checkpoint_dir)
        self.epochs_so_far = 0
    
    def train(self, epochs, checkpoint=False, train_loader=None, val_loader=None):
        # Phases and Logging
        phases = { 'train': train_loader if train_loader else self.train_loader, 
                   'val': val_loader if val_loader else self.val_loader }
        start_time = time.time()
        train_log = []
        
        # Training
        for i in range(epochs):
            epoch_data = { 'train_mean_loss': 0.0, 'val_mean_loss': 0.0 }
            for phase, loader in phases.items():
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                running_loss = 0.0
                total, correct = 0, 0
                for batch in tqdm(loader):
                    _in, _out, _mask = batch[self.in_key].to(self.device), batch[self.target_key].to(self.device), batch[self.mask_key].to(self.device)
                    
                    # Forward
                    self.optimizer.zero_grad()
                    output = self.net(_in)
                    
                    # Apply loss to masked outputs
                    output, _out = output.permute(0, 2, 3, 1), _out.permute(0, 2, 3, 1)
                    _mask = _mask.squeeze()
                    output, _out = output[_mask != 0].float(), _out[_mask != 0].float()
                    loss = self.loss(output, _out)
                    
                    # Optimize
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                        
                    # Log batch results
                    running_loss += loss.item()
                    torch.cuda.empty_cache()
                
                # Log phase results
                epoch_data[phase + '_mean_loss'] = running_loss / len(loader)

            # Display Progress
            duration_elapsed = time.time() - start_time
            print('\n-- Finished Epoch {}/{} --'.format(i, epochs - 1))
            print('Training Loss: {}'.format(epoch_data['train_mean_loss']))
            print('Validation Loss: {}'.format(epoch_data['val_mean_loss']))
            print('Time since start: {}'.format(duration_elapsed))
            epoch_data['time_elapsed'] = duration_elapsed
            train_log.append(epoch_data)
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Checkpoint
            checkpoint_time = time.time()
            if checkpoint:
                path = self.checkpoint_dir + 'checkpoint_' + str(self.epochs_so_far) + '_' + str(checkpoint_time)
                torch.save(self.net.state_dict(), path)
            self.epochs_so_far += 1
            
            # Save train_log
            path = self.checkpoint_dir + 'train_log_' + str(checkpoint_time) 
            with open(path, 'wb') as fp:
                pickle.dump(train_log, fp)
                
        return train_log
    
    def evaluate(self, batch):
        self.net.eval()
        _in = batch[self.in_key].to(self.device)
        output = self.net(_in)
        return output.cpu().detach().numpy()
    
class DualMaskTrainer:
    def __init__(self, device, net, 
                 train_set, 
                 val_set,  
                 batch_size, 
                 optimizer, 
                 scheduler=None,
                 losses = [nn.CrossEntropyLoss(), nn.MSELoss()], 
                 alpha = 0.5,
                 in_key = 0, 
                 target_key = 1,
                 mask_key = 2,
                 num_workers=0,
                 checkpoint_dir='./models/', 
                 exp_name='net'):
        self.device = device
        self.net = net.to(device)
        self.losses = losses
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                       num_workers=num_workers, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, 
                                     num_workers=num_workers, shuffle=False)
        self.in_key = in_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.optimizer = optimizer
        self.scheduler = scheduler
        logging_dir_name = exp_name + '_' + str(time.time()) + '/'
        self.checkpoint_dir = checkpoint_dir + logging_dir_name
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        os.mkdir(self.checkpoint_dir)
        self.epochs_so_far = 0
    
    def train(self, epochs, checkpoint=False, train_loader=None, val_loader=None):
        loss_seg, loss_graph = self.losses
        
        # Phases and Logging
        phases = { 'train': train_loader if train_loader else self.train_loader, 
                   'val': val_loader if val_loader else self.val_loader }
        start_time = time.time()
        train_log = []
        
        # Training
        for i in range(epochs):
            epoch_data = { 'train_mean_loss_seg': 0.0, 'train_mean_loss_graph': 0.0,
                           'val_mean_loss_seg': 0.0, 'val_mean_loss_graph': 0.0 }
            for phase, loader in phases.items():
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                running_losses = np.zeros(2)
                total, correct = 0, 0
                for batch in tqdm(loader):
                    _in, _out, _mask = batch[self.in_key].to(self.device), batch[self.target_key], batch[self.mask_key].to(self.device)
                    _out_seg, _out_graph = _out
                    _out_seg, _out_graph = _out_seg.to(self.device).squeeze().long(), _out_graph.to(self.device)
                    
                    # Forward
                    self.optimizer.zero_grad()
                    output_seg, output_graph = self.net(_in)
                    
                    # Apply loss to masked outputs
                    output_graph, _out_graph = output_graph.permute(0, 2, 3, 1), _out_graph.permute(0, 2, 3, 1)
                    _mask = _mask.squeeze()
                    output_graph, _out_graph = output_graph[_mask != 0].float(), _out_graph[_mask != 0].float()
                    
                    loss0, loss1 = loss_seg(output_seg, _out_seg), loss_graph(output_graph, _out_graph)
                    loss = self.alpha * loss0 + (1 - self.alpha) * loss1
                    
                    # Optimize
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                        
                    # Log batch results
                    running_losses += [loss0.item(), loss1.item()]
                    torch.cuda.empty_cache()
                
                # Log phase results
                running_loss_seg, running_loss_graph = running_losses
                epoch_data[phase + '_mean_loss_seg'] = running_loss_seg / len(loader)
                epoch_data[phase + '_mean_loss_graph'] = running_loss_graph / len(loader)

            # Display Progress
            duration_elapsed = time.time() - start_time
            print('\n-- Finished Epoch {}/{} --'.format(i, epochs - 1))
            print('Training Loss (Segmentation): {}'.format(epoch_data['train_mean_loss_seg']))
            print('Training Loss (Graph): {}'.format(epoch_data['train_mean_loss_graph']))
            print('Validation Loss (Segmentation): {}'.format(epoch_data['val_mean_loss_seg']))
            print('Validation Loss (Graph): {}'.format(epoch_data['val_mean_loss_graph']))
            print('Time since start: {}'.format(duration_elapsed))
            epoch_data['time_elapsed'] = duration_elapsed
            train_log.append(epoch_data)
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Checkpoint
            checkpoint_time = time.time()
            if checkpoint:
                path = self.checkpoint_dir + 'checkpoint_' + str(self.epochs_so_far) + '_' + str(checkpoint_time)
                torch.save(self.net.state_dict(), path)
            self.epochs_so_far += 1
            
            # Save train_log
            path = self.checkpoint_dir + 'train_log_' + str(checkpoint_time) 
            with open(path, 'wb') as fp:
                pickle.dump(train_log, fp)
                
        return train_log
    
    def evaluate(self, batch):
        self.net.eval()
        _in = batch[self.in_key].to(self.device)
        output_seg, output_graph = self.net(_in)
        return output_seg.cpu().detach().numpy(), output_graph.cpu().detach().numpy()
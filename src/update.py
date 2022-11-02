import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import DatasetSplit

class BenignUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        
    def train(self, net):

        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                log_probs = net(images)
                
                loss = self.loss_func(log_probs, labels.squeeze(dim=-1))

                loss.backward()
                
                optimizer.step()

        return net.state_dict()

class CompromisedUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        
    def train(self, net):

        
        net_freeze = copy.deepcopy(net)
        
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
    
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                log_probs = net(images)
                
                loss = self.loss_func(log_probs, labels.squeeze(dim=-1))

                loss.backward()
                
                optimizer.step()

        for w, w_t in zip(net_freeze.parameters(), net.parameters()):
            w_t.data = (w_t.data - w.data) * self.args.mp_alpha
        
        return net.state_dict()
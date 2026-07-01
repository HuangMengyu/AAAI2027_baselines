import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.TC import TC
# from models.NTXent_loss import *
# from models.loss import loss_ntxent, loss_ntxent_class
# import model_utils.loss as loss

import torch.optim as optim

from tqdm import trange
from scipy.stats import norm
import transformations
import simclr_utitlities

class base_Model(nn.Module): # encoder
    def __init__(self, input_channels=3, final_out_channels = 128, kernel_size = 25, stride = 3, features_len = 127):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = features_len
        # self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # x_flat = x.reshape(x.shape[0], -1)
        # logits = self.logits(x_flat)
        # return logits, x
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, bn = True):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        # print("hidden_channels", hidden_channels)
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.n_classes)
        self.ac = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.ac(hidden)
        # print("hidden", hidden.shape)
        hidden = self.bn(hidden)
        out = self.fc2(hidden)

        return out


class SimCLR(nn.Module):
    def __init__(self, device,
        encoder,
        projector,
        batch_size=32,
        num_epochs=100,
        lr = 1e-4,
        ):
        super(SimCLR, self).__init__()
        self.encoder = encoder

        self.projector = projector

        self.gmp = nn.AdaptiveMaxPool1d(1)    


        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs

        transform_funcs = [
        #     transformations.rotation_transform_vectorized # Use rotation trasnformation
            transformations.scaling_transform_vectorized,
            transformations.negate_transform_vectorized
        ]
        self.transformation_function = simclr_utitlities.generate_composite_transform_function_simple(transform_funcs)

        
        self.optimizer = optim.AdamW([{'params':self.encoder.parameters()},
                                        {'params':self.projector.parameters()},
                                        ], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
    
    def forward(self, train_loader, cut_np=2, cut_fp=3):
        losses = []
        
        pbar = trange(self.num_epochs)
        for e in pbar:  
            for batch in train_loader:
                x = batch   # x.shape:(B, C, T)

                if x.shape[0] < 2:  # the case when there is only one sample in the batch
                    continue

                x = x.permute(0,2,1)  # (B, T, C)

                transform_1 = self.transformation_function(x)
                transform_2 = self.transformation_function(x)

                transform_1 = transform_1.permute(0,2,1) #(B,C,T)
                transform_2 = transform_2.permute(0,2,1) #(B,C,T)

                transform_1 = torch.tensor(transform_1, dtype=torch.float).to(self.device)
                transform_2 = torch.tensor(transform_2, dtype=torch.float).to(self.device)

                transform_1_feat = self.encoder(transform_1)
                transform_2_feat = self.encoder(transform_2)

                transform_1_feat = self.gmp(transform_1_feat).squeeze(2)
                transform_2_feat = self.gmp(transform_2_feat).squeeze(2)

                loss = simclr_utitlities.NT_Xent_loss(transform_1_feat, transform_2_feat, self.device, normalize=True, temperature=1.0, weights=1.0)
                # print(loss.shape)
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))      
            
            self.scheduler.step(e)
            pbar.set_description(str(sum(losses) / len(losses)))


        



class Model(nn.Module):
    def __init__(self, 
        in_dims=3,
        output_dims=127,
        device='cuda',
        num_class=5,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        batch_size=32,
        num_epochs=100,
        lr = 1e-4,
        ):
        super(Model, self).__init__()
        
         
        self.encoder = base_Model(in_dims, final_out_channels, kernel_size, stride, output_dims)
        
        self.projector = nn.Sequential(
            nn.Linear(final_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )
        
        self.SimCLR = SimCLR(device,
            self.encoder,
            self.projector,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr = lr
        )

        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = MLP(128, 64, num_class)
        # self.classifier = MLP(128 * 2, 128, num_class)

    def forward(self, x, ssl = False):
        if ssl == False: # input will be a single sample
            features = self.encoder(x)
            features = self.gap(features).squeeze(2)
            return self.classifier(features)
        return self.SimCLR(x) #input will be a dataloader
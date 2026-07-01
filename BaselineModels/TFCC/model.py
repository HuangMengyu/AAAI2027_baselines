import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.NTXent_loss import *

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

class TFCC(nn.Module):
    def __init__(self, device,
        time_encoder,
        input_channels=3, 
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        features_len = 127, 
        timesteps=50, 
        hidden_dim=64,
        batch_size=32
        ):
        super(TFCC, self).__init__()
        # self.time_encoder = base_Model(input_channels, final_out_channels, kernel_size, stride, features_len)
        self.time_encoder = time_encoder
        # self.freq_encoder = base_Model(input_channels, final_out_channels, kernel_size, stride, features_len)
        self.freq_encoder = time_encoder  # let time and freq encoder share weights
        self.temporal_contrasting = TC(device, final_out_channels, timesteps, hidden_dim)
        # self.frequency_contrasting = TC(device, final_out_channels, timesteps, hidden_dim)
        self.frequency_contrasting = self.temporal_contrasting  # let time and freq contrasting share weights
        self.batch_size = batch_size
        self.device = device
    
    def forward(self, x):
        time, freq, time_aug, freq_aug = x[0], x[1], x[2], x[3]
        time_feat = self.time_encoder(time)
        time_aug_feat = self.time_encoder(time_aug)
        freq_feat = self.freq_encoder(freq)
        freq_aug_feat = self.freq_encoder(freq_aug)

        time_feat = F.normalize(time_feat, dim=1)
        time_aug_feat = F.normalize(time_aug_feat, dim=1)
        freq_feat = F.normalize(freq_feat, dim=1)
        freq_aug_feat = F.normalize(freq_aug_feat, dim=1)

        L_T, c_T = self.temporal_contrasting(time_feat, time_aug_feat)
        L_T_aug, c_T_aug = self.temporal_contrasting(time_aug_feat, time_feat)

        L_F, c_F = self.frequency_contrasting(freq_feat, freq_aug_feat)
        L_F_aug, c_F_aug = self.frequency_contrasting(freq_aug_feat, freq_feat)


        nt_xent_criterion = NTXentLoss_poly(self.device, self.batch_size, 0.2, True) # device, 128, 0.2, True

        L_C_T = nt_xent_criterion(c_T, c_T_aug)
        L_C_F = nt_xent_criterion(c_F, c_F_aug)

        # print("L_C_T", L_C_T.shape)
 
        l_TF = F.cosine_similarity(c_T, c_F, dim=1)
        l_1, l_2, l_3 = F.cosine_similarity(c_T, c_F_aug, dim=1), F.cosine_similarity(c_T_aug, c_F, dim=1), F.cosine_similarity(c_T_aug, c_F_aug, dim=1)
        
        L_C_TF = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        L_C_TF = torch.mean(L_C_TF)
        # print("L_C_TF", L_C_TF.shape)

        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4

        loss_TFCC = lambda_1*(L_T+L_T_aug+L_F+L_F_aug) + lambda_2*(L_C_T+L_C_F) + lambda_3*L_C_TF

        return loss_TFCC

        



class Model(nn.Module):
    def __init__(self, 
        input_dims,
        output_dims=127,
        device='cuda',
        num_class=5,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        timesteps=50, 
        hidden_dim=64, 
        batch_size=32):
        super(Model, self).__init__()
        self.encoder = base_Model(input_dims, final_out_channels, kernel_size, stride, output_dims)
        self.TFCC = TFCC(device,
        time_encoder = self.encoder,
        input_channels=input_dims, 
        final_out_channels = final_out_channels, 
        kernel_size = kernel_size, 
        stride = stride, 
        features_len = output_dims, 
        timesteps=timesteps, 
        hidden_dim=hidden_dim,
        batch_size=batch_size
        )
        MLP_input_dim = output_dims * final_out_channels
        # self.classifier = MLP(MLP_input_dim, MLP_input_dim // 2, num_class)
        self.classifier = MLP(hidden_dim, hidden_dim // 2, num_class)



    def forward(self, x, ssl = False):
        if ssl == False:
            features = self.encoder(x)
            transformer_features = self.TFCC.temporal_contrasting.seq_transformer(features.transpose(1,2)).reshape(x.shape[0], -1)
            return self.classifier(transformer_features)
        return self.TFCC(x)
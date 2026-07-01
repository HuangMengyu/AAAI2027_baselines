import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from ts2vec import TS2Vec

class base_Model(nn.Module): # encoder # change to keep the same time steps for output
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

# class base_Model(nn.Module):
#     def __init__(self, input_channels=3, final_out_channels=128, kernel_size=25):
#         super(base_Model, self).__init__()

#         # Block 1: Removed stride=3, added dilation=1
#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
#                       stride=1, bias=False, padding='same'), # 'same' keeps T constant
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             # nn.MaxPool1d removed (would halve T)
#             nn.Dropout(0.35)
#         )

#         # Block 2: Added dilation=2 to expand receptive field
#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, 
#                       padding='same', dilation=2), 
#             nn.BatchNorm1d(64),
#             nn.ReLU()
#             # nn.MaxPool1d removed
#         )

#         # Block 3: Added dilation=4 to expand receptive field further
#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, 
#                       padding='same', dilation=4),
#             nn.BatchNorm1d(final_out_channels),
#             nn.ReLU()
#             # nn.MaxPool1d removed
#         )

#     def forward(self, x_in):
#         x = self.conv_block1(x_in)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, bn = True):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.n_classes)
        self.ac = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.ac(hidden)
        hidden = self.bn(hidden)
        out = self.fc2(hidden)

        return out

class Model(nn.Module):
    def __init__(self, 
        input_dims,
        output_dims=127,
        depth=10,
        device='cuda',
        num_class=5,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        batch_size=32,
        num_epochs=100,
        lr = 1e-4):
        super(Model, self).__init__()

        # self.encoder = base_Model(input_dims, final_out_channels, kernel_size, stride, output_dims)
        self.encoder = base_Model(input_dims, final_out_channels, kernel_size)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.ts2vec = TS2Vec(
                encoder = self.encoder,
                input_dims=input_dims,
                device=device,
                lr=lr,
                batch_size=batch_size)
        self.classifier = MLP(final_out_channels, final_out_channels//2, num_class)

    def forward(self, x, ssl = False, ratio = 0.5):
        if ssl == False:
            features = self.encoder(x)
            features = self.gap(features).squeeze(2)
            return self.classifier(features)
        return self.ts2vec(x)
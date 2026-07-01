import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import signal

# from train_utils.model_selection import init_pretrain_framework, init_loss_func
# from train_utils.loss_calc_utils import calc_pretrain_loss
# from general_utils.weight_utils import freeze_patch_embedding
from data_augmenter.Augmenter import Augmenter
from models.loss import FOCALLoss

import torch.optim as optim

from tqdm import trange
from scipy.stats import norm

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

class FOCAL_main(nn.Module):
    def __init__(self,
        args, 
        device,
        encoder1,
        encoder2,
        batch_size=32,
        num_epochs=100,
        lr = 1e-4,
        ):
        super(FOCAL_main, self).__init__()
        self.EEG_encoder = encoder1
        self.EOG_encoder = encoder2
        self.encoder = [self.EEG_encoder, self.EOG_encoder]

        self.args = args
        
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs


        # self.default_model = init_pretrain_framework(args, self.encoder)
        # self.default_model = freeze_patch_embedding(args, self.default_model)

        self.augmenter = Augmenter(args)
        self.augmenter.to(self.device)
        
        self.loss_func = FOCALLoss(args).to(self.device)


        self.optimizer = optim.AdamW([{'params':self.EEG_encoder.parameters()},
                                        {'params':self.EOG_encoder.parameters()},
                                        ], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
    
    def forward(self, train_loader, cut_np=2, cut_fp=3):
        losses = []
        
        pbar = trange(self.num_epochs)
        for e in pbar:  
            # ind = 0
            for batch in train_loader:
                # ind += 1
                # if ind == 10:
                #     break
                EEG, EOG = batch
                if EEG.shape[0] < 2:  # the case when there is only one sample in the batch
                    continue
                # print("index", index)
                # EEG = EEG.to(self.device)
                # EOG = EOG.to(self.device)

                B, C_1, T = EEG.shape  # [Batch, Channels, 128]
                B, C_2, T = EOG.shape  # [Batch, Channels, 128]
                
                # # Calculate S and I
                # S = int(math.sqrt(2 * T))
                # I = T // S  # Ensuring integer division
                # valid_T = I * S
                # # Reshape to 4D [B, C, I, S]
                # EEG = EEG[:, :, :valid_T].view(B, C_1, I, S)
                # EOG = EOG[:, :, :valid_T].view(B, C_2, I, S)

                # use stft
                # 1. Reshape to 2D: (B*C, T)
                EEG_reshaped = EEG.reshape(-1, T)
                EOG_reshaped = EOG.reshape(-1, T)

                # 2. Apply STFT
                # f: frequencies, t: time segments, Zxx: STFT result
                _, _, Zxx_EEG = signal.stft(EEG_reshaped, nperseg=64)
                _, _, Zxx_EOG = signal.stft(EOG_reshaped, nperseg=64)

                # Zxx shape is currently (B*C, n_frequencies, n_time_intervals)
                # We want: (B, C, n_time_intervals, n_frequencies)

                # 3. Transpose to (B*C, n_time_intervals, n_frequencies)
                Zxx_EEG = np.transpose(Zxx_EEG, (0, 2, 1))
                Zxx_EOG = np.transpose(Zxx_EOG, (0, 2, 1))

                # 4. Reshape back to (B, C, I, S)
                I_EEG = Zxx_EEG.shape[1] # Number of intervals
                S_EEG = Zxx_EEG.shape[2] # Spectrum length (freq bins)
                final_output_EEG = Zxx_EEG.reshape(B, C_1, I_EEG, S_EEG)

                I_EOG = Zxx_EOG.shape[1] # Number of intervals
                S_EOG = Zxx_EOG.shape[2] # Spectrum length (freq bins)
                final_output_EOG = Zxx_EOG.reshape(B, C_2, I_EOG, S_EOG)

                                
                # retrieve the features from two modalities
                # assume that mod_features1 and mod_features2 are two lists of modality features
                # looks like this: [(b,c,dim), (b,c,dim)]
                
                final_output_EEG = torch.tensor(final_output_EEG).to(self.device)
                final_output_EOG = torch.tensor(final_output_EOG).to(self.device)

                x = {"mod1": final_output_EEG, "mod2": final_output_EOG}  # time domain inputs    
                
                
                aug_freq_loc_inputs_1 = self.augmenter.forward("random", x)
                aug_freq_loc_inputs_2 = self.augmenter.forward("random", x)

                feature1 = dict()
                feature2 = dict()
                # print(aug_freq_loc_inputs_1["mod1"].shape, aug_freq_loc_inputs_1["mod2"].shape)
                # print(aug_freq_loc_inputs_2["mod1"].shape, aug_freq_loc_inputs_2["mod2"].shape)

                B,C_1,d1_1,d2_1 = aug_freq_loc_inputs_1["mod1"].shape
                B,C_2,d1_2,d2_2 = aug_freq_loc_inputs_1["mod2"].shape
                B,C_3,d1_3,d2_3 = aug_freq_loc_inputs_2["mod1"].shape
                B,C_4,d1_4,d2_4 = aug_freq_loc_inputs_2["mod2"].shape

                feature1["mod1"] = self.EEG_encoder(aug_freq_loc_inputs_1["mod1"].reshape(B, C_1, -1))
                feature1["mod2"] = self.EOG_encoder(aug_freq_loc_inputs_1["mod2"].reshape(B, C_2, -1))
                feature2["mod1"] = self.EEG_encoder(aug_freq_loc_inputs_2["mod1"].reshape(B, C_3, -1))
                feature2["mod2"] = self.EOG_encoder(aug_freq_loc_inputs_2["mod2"].reshape(B, C_4, -1))

                loss = self.loss_func(feature1, feature2)
        

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))      
            
            self.scheduler.step(e)
            pbar.set_description(str(sum(losses) / len(losses)))


        



class Model(nn.Module):
    def __init__(self, 
        args,
        mod1_dims=2,
        mod2_dims=1,
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
        
        # self.EEG_dims = EEG_dims
        # self.EOG_dims = EOG_dims

        self.EEG_encoder = base_Model(mod1_dims * 2, final_out_channels, kernel_size, stride, output_dims)
        self.EOG_encoder = base_Model(mod2_dims * 2, final_out_channels, kernel_size, stride, output_dims)
        
        self.focal = FOCAL_main(args,
        device,
        self.EEG_encoder,
        self.EOG_encoder,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr = lr,
        )
        
        self.augmenter = Augmenter(args)
        self.gap = nn.AdaptiveAvgPool1d(1)

        
        self.classifier = MLP(128 * 2, 128, num_class)

    def forward(self, x, ssl = False):
        if ssl == False: # input will be a single sample
            EEG, EOG = x[0], x[1]

            B, C_1, T = EEG.shape  # [Batch, Channels, 128]
            B, C_2, T = EOG.shape  # [Batch, Channels, 128]
            # Calculate S and I
            S = int(math.sqrt(2 * T))
            I = T // S  # Ensuring integer division
            valid_T = I * S

            # Reshape to 4D [B, C, I, S]
            EEG = EEG[:, :, :valid_T].view(B, C_1, I, S)
            EOG = EOG[:, :, :valid_T].view(B, C_2, I, S)
            # print(EEG.shape, EOG.shape)
            
            # do the fft preprocess as in pretraining
            x = {"mod1": EEG, "mod2": EOG}

            x = self.augmenter.fft_preprocess(x)

            EEG_features = self.EEG_encoder(x["mod1"].reshape(B, 2*C_1, -1))
            EEG_features = self.gap(EEG_features)
            # EEG_features = self.projector(EEG_features)
            EOG_features = self.EOG_encoder(x["mod2"].reshape(B, 2*C_2, -1))
            EOG_features = self.gap(EOG_features)
            # EOG_features = self.projector(EOG_features)

            # print(EEG_features.shape, EOG_features.shape)

            EEG_features_flat = EEG_features.reshape(EEG.shape[0], -1)
            EOG_features_flat = EOG_features.reshape(EOG.shape[0], -1)
       
            features_flat = torch.cat((EEG_features_flat, EOG_features_flat), dim=1)
       
            return self.classifier(features_flat)
        return self.focal(x) #input will be a dataloader
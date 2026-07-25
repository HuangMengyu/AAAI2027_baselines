import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.NTXent_loss import *
from models.loss import loss_ntxent, loss_ntxent_class
from models.fnd import LambdaThreshold

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

class TFCC(nn.Module):
    def __init__(self, device,
        EEG_encoder,
        EOG_encoder,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        features_len = 127, 
        timesteps=50, 
        hidden_dim=64,
        batch_size=32,
        num_epochs=100,
        warm_epochs=50,
        lr = 1e-4,
        use_iteration=False,
        num_iter=600,
        depth=4,
        glofnd='none',
        glofnd_data_size=1,
        glofnd_alpha=0.01,
        glofnd_lr_lda=0.05,
        glofnd_start_update=0,
        glofnd_start_lda=0,
        glofnd_init_quantile=False,
        glofnd_clip_grad_mult=None,
        ):
        super(TFCC, self).__init__()
        self.EEG_encoder = EEG_encoder
        self.EOG_encoder = EOG_encoder  
        self.EEG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
        self.EOG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        self.warm_epochs = warm_epochs
        self.device = device

        self.shared_projection_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.shared_projection_head_2 = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.optimizer = optim.AdamW([{'params':self.EEG_encoder.parameters()},
                                        {'params':self.EOG_encoder.parameters()},
                                        {'params':self.EEG_contrasting.parameters()},
                                        {'params':self.EOG_contrasting.parameters()},
                                        {'params':self.shared_projection_head_1.parameters()},
                                        {'params':self.shared_projection_head_2.parameters()}], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
    
        self.use_iteration = use_iteration
        self.num_iter = num_iter

        self.aux_classifier = MLP(final_out_channels // 4, final_out_channels // 8, 5)
        self.glofnd = glofnd
        self.glofnd_data_size = max(int(glofnd_data_size), 1)

        def build_lambda_threshold():
            return LambdaThreshold(
                glofnd_type=glofnd,
                N=self.glofnd_data_size,
                alpha=glofnd_alpha,
                lr_lda=glofnd_lr_lda,
                start_update=glofnd_start_update,
                lda_start=glofnd_start_lda,
                init_quantile=glofnd_init_quantile,
                clip_grad_mult=glofnd_clip_grad_mult,
            )

        self.lambda_temporal_eeg = build_lambda_threshold()
        self.lambda_temporal_eog = build_lambda_threshold()
        self.lambda_context_t = build_lambda_threshold()
        self.lambda_context_f = build_lambda_threshold()
        self.lambda_context_tf = build_lambda_threshold()


    def forward(self, train_loader, cut_np=2, cut_fp=3):

        # torch.autograd.set_detect_anomaly(True)
        
        #* indicies of instance in dataset
        indicies = np.arange(len(train_loader.dataset))
        #
        losses = []
        # torch.autograd.set_detect_anomaly(True)

        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4

        
        pbar = trange(self.num_epochs)
        for e in pbar:
            
            
            if not self.use_iteration:
                
                for batch in train_loader:
                    EEG, EOG, EEG_aug, EOG_aug, index = batch
                    if index.shape[0] < 2:  # the case when there is only one sample in the batch
                        continue
                    # print("index", index)
                    EEG = EEG.to(self.device)
                    EOG = EOG.to(self.device)
                    EEG_aug = EEG_aug.to(self.device)
                    EOG_aug = EOG_aug.to(self.device)
                    x = (EEG, EOG, EEG_aug, EOG_aug)

                    
                    sum_1, L_C_T, L_C_F, L_C_TF, pos_sim_T, pos_sim_F, pos_sim_TF, neg_sim_T, neg_sim_F, neg_sim_TF, _, _, _, _, _, _ = self.loss_compute(x, index=index, epoch=e)
                    
                    loss_C_T = torch.mean(L_C_T)
                    loss_C_F = torch.mean(L_C_F)
                    loss_C_TF = torch.mean(L_C_TF)  

                    loss = lambda_1*sum_1 + lambda_2*(loss_C_T + loss_C_F) + lambda_3*loss_C_TF

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss))

                    
                
            self.scheduler.step(e)
            pbar.set_description(str(sum(losses) / len(losses)))


    def loss_compute(self, x, index=None, epoch=None):
        
        EEG, EOG, EEG_aug, EOG_aug = x[0], x[1], x[2], x[3]
        EEG_feat = self.EEG_encoder(EEG)
        EEG_aug_feat = self.EEG_encoder(EEG_aug)
        EOG_feat = self.EOG_encoder(EOG)
        EOG_aug_feat = self.EOG_encoder(EOG_aug)

        EEG_feat = F.normalize(EEG_feat, dim=1)
        EEG_aug_feat = F.normalize(EEG_aug_feat, dim=1)
        EOG_feat = F.normalize(EOG_feat, dim=1)
        EOG_aug_feat = F.normalize(EOG_aug_feat, dim=1)

        L_T, z_T, c_T = self.EEG_contrasting(
            EEG_feat,
            EEG_aug_feat,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_temporal_eeg,
        )
        L_T_aug, _, c_T_aug = self.EEG_contrasting(
            EEG_aug_feat,
            EEG_feat,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_temporal_eeg,
        )

        L_F, z_F, c_F = self.EOG_contrasting(
            EOG_feat,
            EOG_aug_feat,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_temporal_eog,
        )
        L_F_aug, _, c_F_aug = self.EOG_contrasting(
            EOG_aug_feat,
            EOG_feat,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_temporal_eog,
        )

       
        L_C_T, pos_sim_T, neg_sim_T, num_cancel_t, num_attract_t = loss_ntxent(
            [c_T, c_T_aug],
            self.device,
            filter=False,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_context_t,
        )
        L_C_F, pos_sim_F, neg_sim_F, num_cancel_f, num_attract_f = loss_ntxent(
            [c_F, c_F_aug],
            self.device,
            filter=False,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_context_f,
        )


        
        ############################ Compute Contrastive Loss Instead ##############################

        c_T_shared = self.shared_projection_head_1(z_T)
        c_F_shared = self.shared_projection_head_2(z_F)

        L_C_TF_1, pos_sim_TF_1, neg_sim_TF_1, num_cancel_tf_1, num_attract_tf_1 = loss_ntxent(
            [c_T_shared, c_F_shared],
            self.device,
            filter=False,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_context_tf,
        )
        L_C_TF_2, pos_sim_TF_2, neg_sim_TF_2, num_cancel_tf_2, num_attract_tf_2 = loss_ntxent(
            [c_F_shared, c_T_shared],
            self.device,
            filter=False,
            index=index,
            epoch=epoch,
            lambda_threshold=self.lambda_context_tf,
        )
        num_cancel_tf = num_cancel_tf_1 + num_cancel_tf_2
        num_attract_tf = num_attract_tf_1 + num_attract_tf_2
        
        L_C_TF = (L_C_TF_1 + L_C_TF_2) / 2
        pos_sim_TF = torch.cat((pos_sim_TF_1, pos_sim_TF_2), dim=0)
        neg_sim_TF = torch.cat((neg_sim_TF_1, neg_sim_TF_2), dim=0)
        
        ############################ Compute Contrastive Loss End ##############################

        
        sum_1 = L_T+L_T_aug+L_F+L_F_aug

        return sum_1, L_C_T, L_C_F, L_C_TF, pos_sim_T, pos_sim_F, pos_sim_TF, neg_sim_T, neg_sim_F, neg_sim_TF, num_cancel_t, num_attract_t, num_cancel_f, num_attract_f, num_cancel_tf, num_attract_tf

        



class Model(nn.Module):
    def __init__(self, 
        mod1_dims=2,
        mod2_dims=1,
        output_dims=127,
        device='cuda',
        num_class=5,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        timesteps=50, 
        hidden_dim=64, 
        batch_size=32,
        num_epochs=100,
        warm_epochs=50,
        lr = 1e-4,
        use_iteration=False,
        num_iter=600,
        depth=4,
        glofnd='none',
        glofnd_data_size=1,
        glofnd_alpha=0.01,
        glofnd_lr_lda=0.05,
        glofnd_start_update=0,
        glofnd_start_lda=0,
        glofnd_init_quantile=False,
        glofnd_clip_grad_mult=None,
        **kwargs
        ):
        super(Model, self).__init__()

        self.EEG_encoder = base_Model(mod1_dims, final_out_channels, kernel_size, stride, output_dims)
        self.EOG_encoder = base_Model(mod2_dims, final_out_channels, kernel_size, stride, output_dims)
        
        self.TFCC = TFCC(device,
        EEG_encoder = self.EEG_encoder,
        EOG_encoder = self.EOG_encoder, 
        final_out_channels = final_out_channels, 
        kernel_size = kernel_size, 
        stride = stride, 
        features_len = output_dims, 
        timesteps=timesteps, 
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        warm_epochs=warm_epochs,
        lr = lr,
        use_iteration=use_iteration,
        num_iter=num_iter,
        depth=depth,
        glofnd=glofnd,
        glofnd_data_size=glofnd_data_size,
        glofnd_alpha=glofnd_alpha,
        glofnd_lr_lda=glofnd_lr_lda,
        glofnd_start_update=glofnd_start_update,
        glofnd_start_lda=glofnd_start_lda,
        glofnd_init_quantile=glofnd_init_quantile,
        glofnd_clip_grad_mult=glofnd_clip_grad_mult,
        )
        
        self.classifier = MLP(hidden_dim * 2, hidden_dim, num_class)
        

    def forward(self, x, ssl = False):
        if ssl == False: # input will be a single sample
            EEG, EOG = x[0], x[1]
            EEG_features = self.EEG_encoder(EEG)
            EOG_features = self.EOG_encoder(EOG)


            EEG_transformer_features = self.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2)).reshape(EEG.shape[0], -1)
            EOG_transformer_features = self.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2)).reshape(EOG.shape[0], -1)
            features_flat = torch.cat((EEG_transformer_features, EOG_transformer_features), dim=1)
           
            return self.classifier(features_flat)
        return self.TFCC(x) #input will be a dataloader

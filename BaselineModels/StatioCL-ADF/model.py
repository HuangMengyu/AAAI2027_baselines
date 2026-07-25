import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.NTXent_loss import *
from models.loss import loss_ntxent, loss_ntxent_adf

import torch.optim as optim

from tqdm import trange

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
        depth=4
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

    def _train_one_batch(self, batch, use_adf):
        EEG, EOG, EEG_aug, EOG_aug, sample_adf, segment_adf, index = batch
        if index.shape[0] < 2:
            return None, None
        EEG = EEG.to(self.device)
        EOG = EOG.to(self.device)
        EEG_aug = EEG_aug.to(self.device)
        EOG_aug = EOG_aug.to(self.device)
        sample_adf = sample_adf.to(self.device)
        segment_adf = segment_adf.to(self.device)
        x = (EEG, EOG, EEG_aug, EOG_aug)

        sum_1, L_C_T, L_C_F, L_C_TF, stats = self.loss_compute(
            x,
            sample_adf=sample_adf,
            segment_adf=segment_adf,
            use_adf=use_adf,
        )

        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4
        loss = lambda_1 * sum_1 + lambda_2 * (L_C_T + L_C_F) + lambda_3 * L_C_TF

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, stats

    def forward(self, train_loader, cut_np=2, cut_fp=3):

        losses = []
        ratio_sums = {
            "sample": 0.0,
            "segment": 0.0,
        }
        ratio_count = 0

        if self.use_iteration:
            pbar = trange(self.num_iter)
            data_iter = iter(train_loader)
            for step in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                use_adf = step >= self.warm_epochs
                loss, stats = self._train_one_batch(batch, use_adf=use_adf)
                if loss is None:
                    continue
                losses.append(float(loss))
                ratio_sums["sample"] += stats["sample_valid_negative_ratio"]
                ratio_sums["segment"] += stats["segment_valid_negative_ratio"]
                ratio_count += 1
                self.scheduler.step(float(loss))
                avg_loss = sum(losses) / len(losses)
                pbar.set_description(
                    f"phase={'adf' if use_adf else 'warmup'} loss={avg_loss:.4f} "
                    f"sample_neg={ratio_sums['sample']/ratio_count:.4f} "
                    f"segment_neg={ratio_sums['segment']/ratio_count:.4f}"
                )
            return

        pbar = trange(self.num_epochs)
        for e in pbar:
            use_adf = e >= self.warm_epochs
            for batch in train_loader:
                loss, stats = self._train_one_batch(batch, use_adf=use_adf)
                if loss is None:
                    continue
                losses.append(float(loss))
                ratio_sums["sample"] += stats["sample_valid_negative_ratio"]
                ratio_sums["segment"] += stats["segment_valid_negative_ratio"]
                ratio_count += 1
            if losses:
                self.scheduler.step(sum(losses) / len(losses))
                pbar.set_description(
                    f"phase={'adf' if use_adf else 'warmup'} "
                    f"loss={sum(losses)/len(losses):.4f} "
                    f"sample_neg={ratio_sums['sample']/max(ratio_count, 1):.4f} "
                    f"segment_neg={ratio_sums['segment']/max(ratio_count, 1):.4f}"
                )

    def loss_compute(self, x, sample_adf, segment_adf, use_adf=True):
        
        EEG, EOG, EEG_aug, EOG_aug = x[0], x[1], x[2], x[3]
        EEG_feat = self.EEG_encoder(EEG)
        EEG_aug_feat = self.EEG_encoder(EEG_aug)
        EOG_feat = self.EOG_encoder(EOG)
        EOG_aug_feat = self.EOG_encoder(EOG_aug)

        EEG_feat = F.normalize(EEG_feat, dim=1)
        EEG_aug_feat = F.normalize(EEG_aug_feat, dim=1)
        EOG_feat = F.normalize(EOG_feat, dim=1)
        EOG_aug_feat = F.normalize(EOG_aug_feat, dim=1)

        L_T, z_T, c_T, seg_ratio_t = self.EEG_contrasting(
            EEG_feat,
            EEG_aug_feat,
            segment_adf=segment_adf if use_adf else None,
        )
        L_T_aug, _, c_T_aug, seg_ratio_t_aug = self.EEG_contrasting(
            EEG_aug_feat,
            EEG_feat,
            segment_adf=segment_adf if use_adf else None,
        )

        L_F, z_F, c_F, seg_ratio_f = self.EOG_contrasting(
            EOG_feat,
            EOG_aug_feat,
            segment_adf=segment_adf if use_adf else None,
        )
        L_F_aug, _, c_F_aug, seg_ratio_f_aug = self.EOG_contrasting(
            EOG_aug_feat,
            EOG_feat,
            segment_adf=segment_adf if use_adf else None,
        )

       
        if use_adf:
            L_C_T, _, _, ratio_t = loss_ntxent_adf([c_T, c_T_aug], sample_adf, self.device)
            L_C_F, _, _, ratio_f = loss_ntxent_adf([c_F, c_F_aug], sample_adf, self.device)
        else:
            L_C_T, _, _, _, _ = loss_ntxent([c_T, c_T_aug], self.device, filter=False)
            L_C_F, _, _, _, _ = loss_ntxent([c_F, c_F_aug], self.device, filter=False)
            batch_size = c_T.shape[0]
            ratio_t = ratio_f = float(batch_size - 1) / float(batch_size)


        
        ############################ Compute Contrastive Loss Instead ##############################

        c_T_shared = self.shared_projection_head_1(z_T)
        c_F_shared = self.shared_projection_head_2(z_F)

        if use_adf:
            L_C_TF_1, _, _, ratio_tf_1 = loss_ntxent_adf([c_T_shared, c_F_shared], sample_adf, self.device)
            L_C_TF_2, _, _, ratio_tf_2 = loss_ntxent_adf([c_F_shared, c_T_shared], sample_adf, self.device)
        else:
            L_C_TF_1, _, _, _, _ = loss_ntxent([c_T_shared, c_F_shared], self.device, filter=False)
            L_C_TF_2, _, _, _, _ = loss_ntxent([c_F_shared, c_T_shared], self.device, filter=False)
            batch_size = c_T_shared.shape[0]
            ratio_tf_1 = ratio_tf_2 = float(batch_size - 1) / float(batch_size)
        
        L_C_TF = (L_C_TF_1 + L_C_TF_2) / 2
        
        ############################ Compute Contrastive Loss End ##############################

        
        sum_1 = L_T+L_T_aug+L_F+L_F_aug
        stats = {
            "sample_valid_negative_ratio": (ratio_t + ratio_f + ratio_tf_1 + ratio_tf_2) / 4.0,
            "segment_valid_negative_ratio": (
                seg_ratio_t + seg_ratio_t_aug + seg_ratio_f + seg_ratio_f_aug
            ) / 4.0,
        }

        return sum_1, L_C_T, L_C_F, L_C_TF, stats

        



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
        cut_off=None,
        attract_filter=None,
        small_classifier=False
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
        depth=depth
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

import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.NTXent_loss import *
from models.loss import loss_ntxent

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
        option = 'bp_A',
        topk = 0.25,
        fn_mode = 'cancel',
        use_iteration=False,
        num_iter=600
        ):
        super(TFCC, self).__init__()
        self.EEG_encoder = EEG_encoder
        self.EOG_encoder = EOG_encoder  
        self.EEG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim)
        self.EOG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim)
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        self.warm_epochs = warm_epochs
        self.device = device

        self.optimizer = optim.AdamW([{'params':self.EEG_encoder.parameters()},
                                        {'params':self.EOG_encoder.parameters()},
                                        {'params':self.EEG_contrasting.parameters()},
                                        {'params':self.EOG_contrasting.parameters()}], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
    
        self.option = option
        self.removal_option, self.position_option = option.split('_')[0], option.split('_')[1]
        # print(self.removal_option, self.position_option)

        self.topk = topk
        self.fn_mode = fn_mode
        # print('self.fn_mode:', self.fn_mode)

        self.use_iteration = use_iteration
        self.num_iter = num_iter
    
    def index_np_ft(self, epoch_weight, memory_module, cut_np=2, cut_fp=3):
        instance_mean = {k: np.mean(np.array(v)*epoch_weight) \
                            for k, v in sorted(memory_module.items(), key=lambda item: item[1]) \
                            if len(v) == len(epoch_weight)}
        
        mu = np.mean(list(instance_mean.values()))
        sd = np.std(list(instance_mean.values()))

        #* global statistic
        gaussian_norm = norm(mu, sd)

        #* thresholds for noisy positive pairs and faulty positive pairs
        np_bound = mu-cut_np*sd
        fp_bound = mu+cut_fp*sd

        #* identify potential bad positive pairs
        np_index = [k for k in instance_mean.keys() if instance_mean[k]<=np_bound]
        fp_index = [k for k in instance_mean.keys() if instance_mean[k]>=fp_bound]

        return np_index, fp_index, gaussian_norm
    
    def bad_positive_reweighting(self, L, index, np_index, fp_index, gaussian_norm):
        l = L.detach().cpu()
        w = gaussian_norm.pdf(l)
        for i in range(len(index)):
            _id = index[i].cpu().item()
            if _id in np_index:
                #* penalize potential np
                L[i] *= w[i]
            elif _id in fp_index:
                #* penalize potential fp  
                L[i] *= w[i]
        return L

    def forward(self, train_loader, cut_np=2, cut_fp=3):
        
        #* indicies of instance in dataset
        indicies = np.arange(len(train_loader.dataset))
        #* dictionary to record instance training loss (the memory module)
        if self.removal_option in ['bp', 'bpfn'] and self.position_option in ['A', 'AB']:
            memory_module_T = dict([(k,[]) for k in indicies])
            memory_module_F = dict([(k,[]) for k in indicies])
        if self.removal_option in ['bp', 'bpfn'] and self.position_option in ['B', 'AB']:
            memory_module_TF = dict([(k,[]) for k in indicies])

        losses = []
        # torch.autograd.set_detect_anomaly(True)

        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4
        
        if self.use_iteration:
            pbar = trange(self.num_iter)
            train_iter = iter(train_loader)
            self.warm_epochs = int(self.num_iter // 2)
        else:
            pbar = trange(self.num_epochs)
        
        for e in pbar:
            if self.removal_option in ['bp', 'bpfn', 'fn'] and e == self.warm_epochs:
                print("Epoch {e} Warmup stage finished. Start reweighting. Option: {self.removal_option}, {self.position_option}")
            
            if not self.use_iteration:
                epoch_weight = [(1+i)/self.num_epochs for i in range(e)]
            else: 
                epoch_weight = [(1+i)/self.num_iter for i in range(e)]
            if e >= self.warm_epochs and self.removal_option in ['bp', 'bpfn'] and self.position_option in ['A', 'AB']: # DBPM reweights after warmup
                print("Updating memory module T, F and calculating bad positive / false negative indices...")
                np_index_t, fp_index_t, gaussian_norm_t = self.index_np_ft(epoch_weight, memory_module_T, cut_np, cut_fp)
                np_index_f, fp_index_f, gaussian_norm_f = self.index_np_ft(epoch_weight, memory_module_F, cut_np, cut_fp)
            if e >= self.warm_epochs and self.removal_option in ['bp', 'bpfn'] and self.position_option in ['B', 'AB']:
                print("Updating memory module TF and calculating bad positive / false negative indices...")
                np_index_tf, fp_index_tf, gaussian_norm_tf = self.index_np_ft(epoch_weight, memory_module_TF, cut_np, cut_fp)
                # print("np_index_t", np_index_t)

            

            if not self.use_iteration:
                for batch in train_loader:
                    EEG, EOG, EEG_aug, EOG_aug, index = batch
                    # print("index", index)
                    EEG = EEG.to(self.device)
                    EOG = EOG.to(self.device)
                    EEG_aug = EEG_aug.to(self.device)
                    EOG_aug = EOG_aug.to(self.device)
                    x = (EEG, EOG, EEG_aug, EOG_aug)

                    if e >= self.warm_epochs and self.removal_option in ['fn', 'bpfn']: # False negative cancellation
                        sum_1, L_C_T, L_C_F, L_C_TF = self.loss_compute(x, nf_option = self.position_option, topk=self.topk, fn_mode = self.fn_mode)
                    else:
                        sum_1, L_C_T, L_C_F, L_C_TF = self.loss_compute(x, nf_option = 'None')

                    #* update the memory module
                    if self.removal_option in ['bp', 'bpfn'] and self.position_option in ['A', 'AB']: # DBPM reweights after warmup
                        for i in range(len(index)):
                            memory_module_T[index[i].cpu().item()].append(L_C_T[i].cpu().item())
                            memory_module_F[index[i].cpu().item()].append(L_C_F[i].cpu().item())
                    if self.removal_option in ['bp', 'bpfn'] and self.position_option in ['B', 'AB']:
                        for i in range(len(index)):
                            memory_module_TF[index[i].cpu().item()].append(L_C_F[i].cpu().item())
                    
                    if e >= self.warm_epochs and self.removal_option in ['bp', 'bpfn'] and self.position_option in ['A', 'AB']: # DBPM reweights after warmup
                        L_C_T = self.bad_positive_reweighting(L_C_T, index, np_index_t, fp_index_t, gaussian_norm_t)
                        L_C_F = self.bad_positive_reweighting(L_C_F, index, np_index_f, fp_index_f, gaussian_norm_f)
                    if e >= self.warm_epochs and self.removal_option in ['bp', 'bpfn'] and self.position_option in ['B', 'AB']:
                        L_C_TF = self.bad_positive_reweighting(L_C_TF, index, np_index_tf, fp_index_tf, gaussian_norm_tf)

                    loss_C_T = torch.mean(L_C_T)
                    loss_C_F = torch.mean(L_C_F)
                    loss_C_TF = torch.mean(L_C_TF)  

                    loss = lambda_1*sum_1 + lambda_2*(loss_C_T + loss_C_F) + lambda_3*loss_C_TF   

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss))
                    # num += 1
                    # if num == 1:
                    #     break
            if self.use_iteration:
                batch = next(train_iter)
                # print("Run Iteration")

                # Same logic as above 
                EEG, EOG, EEG_aug, EOG_aug, index = batch
                # print("index", index)
                EEG = EEG.to(self.device)
                EOG = EOG.to(self.device)
                EEG_aug = EEG_aug.to(self.device)
                EOG_aug = EOG_aug.to(self.device)
                x = (EEG, EOG, EEG_aug, EOG_aug)

                if e >= self.warm_epochs and self.removal_option in ['fn', 'bpfn']: # False negative cancellation
                    sum_1, L_C_T, L_C_F, L_C_TF = self.loss_compute(x, nf_option = self.position_option, topk=self.topk, fn_mode = self.fn_mode)
                else:
                    sum_1, L_C_T, L_C_F, L_C_TF = self.loss_compute(x, nf_option = 'None')

                #* update the memory module
                if self.removal_option in ['bp', 'bpfn'] and self.position_option in ['A', 'AB']: # DBPM reweights after warmup
                    # print("update memory module within batch")
                    for i in range(len(index)):
                        memory_module_T[index[i].cpu().item()].append(L_C_T[i].cpu().item())
                        memory_module_F[index[i].cpu().item()].append(L_C_F[i].cpu().item())
                if self.removal_option in ['bp', 'bpfn'] and self.position_option in ['B', 'AB']:
                    for i in range(len(index)):
                        memory_module_TF[index[i].cpu().item()].append(L_C_F[i].cpu().item())
                
                if e >= self.warm_epochs and self.removal_option in ['bp', 'bpfn'] and self.position_option in ['A', 'AB']: # DBPM reweights after warmup
                    L_C_T = self.bad_positive_reweighting(L_C_T, index, np_index_t, fp_index_t, gaussian_norm_t)
                    L_C_F = self.bad_positive_reweighting(L_C_F, index, np_index_f, fp_index_f, gaussian_norm_f)
                if e >= self.warm_epochs and self.removal_option in ['bp', 'bpfn'] and self.position_option in ['B', 'AB']:
                    L_C_TF = self.bad_positive_reweighting(L_C_TF, index, np_index_tf, fp_index_tf, gaussian_norm_tf)

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


    def loss_compute(self, x, nf_option = 'None', topk=0.25, fn_mode = 'cancel'):
        # print('loss_compute fn_mode:', fn_mode)
        EEG, EOG, EEG_aug, EOG_aug = x[0], x[1], x[2], x[3]
        EEG_feat = self.EEG_encoder(EEG)
        EEG_aug_feat = self.EEG_encoder(EEG_aug)
        EOG_feat = self.EOG_encoder(EOG)
        EOG_aug_feat = self.EOG_encoder(EOG_aug)

        EEG_feat = F.normalize(EEG_feat, dim=1)
        EEG_aug_feat = F.normalize(EEG_aug_feat, dim=1)
        EOG_feat = F.normalize(EOG_feat, dim=1)
        EOG_aug_feat = F.normalize(EOG_aug_feat, dim=1)

        L_T, c_T = self.EEG_contrasting(EEG_feat, EEG_aug_feat)
        L_T_aug, c_T_aug = self.EEG_contrasting(EEG_aug_feat, EEG_feat)

        L_F, c_F = self.EOG_contrasting(EOG_feat, EOG_aug_feat)
        L_F_aug, c_F_aug = self.EOG_contrasting(EOG_aug_feat, EOG_feat)


        # nt_xent_criterion = NTXentLoss_poly(self.device, self.batch_size, 0.2, True) # device, 128, 0.2, True

        # L_C_T = nt_xent_criterion(c_T, c_T_aug)
        # L_C_F = nt_xent_criterion(c_F, c_F_aug)

        if nf_option in ['A', 'AB']:
            # print("Applying false negative cancellation in position A...")
            L_C_T = loss_ntxent([c_T, c_T_aug], self.device, filter=True, topk=topk, fn_mode = fn_mode)
            L_C_F = loss_ntxent([c_F, c_F_aug], self.device, filter=True, topk=topk, fn_mode = fn_mode)
        else:
            L_C_T = loss_ntxent([c_T, c_T_aug], self.device, filter=False)
            L_C_F = loss_ntxent([c_F, c_F_aug], self.device, filter=False)

        # print("L_C_T", L_C_T.shape)
 
        ############################ Compute Triple Loss ##############################
        # l_TF = F.cosine_similarity(c_T, c_F, dim=1)
        # l_1, l_2, l_3 = F.cosine_similarity(c_T, c_F_aug, dim=1), F.cosine_similarity(c_T_aug, c_F, dim=1), F.cosine_similarity(c_T_aug, c_F_aug, dim=1)
        
        # if nf_option in ['B', 'AB']:
        #     # print("Applying false negative cancellation in position B...")
        #     current_batch_size = l_1.shape[0]
        #     topk_value = int(topk * current_batch_size)
        #     topk_value = max(topk_value, 1)
        #     assert topk_value >= 1, "topk_value must be at least 1"

        #     l_1_sorted, _ = torch.sort(l_1, descending=True)
        #     threshold_l_1 = l_1_sorted[topk_value - 1] # threshold value for false negative
            
        #     l_2_sorted, _ = torch.sort(l_2, descending=True)
        #     threshold_l_2 = l_2_sorted[topk_value - 1] # threshold value for false negative

        #     l_3_sorted, _ = torch.sort(l_3, descending=True)
        #     threshold_l_3 = l_3_sorted[topk_value - 1] # threshold value for false negative

        #     if fn_mode == 'cancel':
        #         l_1 = torch.where(l_1 < threshold_l_1, l_1, 0.0) # filter false negatives
        #         l_2 = torch.where(l_2 < threshold_l_2, l_2, 0.0) # filter false negatives
        #         l_3 = torch.where(l_3 < threshold_l_3, l_3, 0.0) # filter false negatives
        #     elif fn_mode == 'attract':
        #         l_1 = torch.where(l_1 > threshold_l_1, l_1, l_TF) # get false negatives
        #         l_2 = torch.where(l_2 > threshold_l_2, l_2, l_TF) # get false negatives
        #         l_3 = torch.where(l_3 > threshold_l_3, l_3, l_TF) # get false negatives

        # L_C_TF = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
        ############################ Compute Triple Loss End ##############################

        ############################ Compute Contrastive Loss Instead ##############################
        if nf_option in ['B', 'AB']:
            L_C_TF = loss_ntxent([c_T, c_F], self.device, filter=True, topk=topk, fn_mode = fn_mode)
        else:
            L_C_TF = loss_ntxent([c_T, c_F], self.device, filter=False)
        ############################ Compute Contrastive Loss End ##############################

        # L_C_TF = torch.mean(L_C_TF)
        # print("L_C_TF", L_C_TF.shape)

        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4

        loss_TFCC = lambda_1*(L_T+L_T_aug+L_F+L_F_aug) + lambda_2*(L_C_T+L_C_F) + lambda_3*L_C_TF

        sum_1 = L_T+L_T_aug+L_F+L_F_aug

        return sum_1, L_C_T, L_C_F, L_C_TF

        



class Model(nn.Module):
    def __init__(self, 
        EEG_dims=2,
        EOG_dims=1,
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
        option = 'bp_A',
        topk = 0.25,
        fn_mode = 'cancel',
        use_iteration=False,
        num_iter=600
        ):
        super(Model, self).__init__()
        
        # self.EEG_dims = EEG_dims
        # self.EOG_dims = EOG_dims

        self.EEG_encoder = base_Model(EEG_dims, final_out_channels, kernel_size, stride, output_dims)
        self.EOG_encoder = base_Model(EOG_dims, final_out_channels, kernel_size, stride, output_dims)
        
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
        option = option,
        topk = topk,
        fn_mode = fn_mode,
        use_iteration=use_iteration,
        num_iter=num_iter
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
            # print("features", features.shape)
            # features_flat = features.reshape(features.shape[0], -1)
            return self.classifier(features_flat)
        return self.TFCC(x) #input will be a dataloader
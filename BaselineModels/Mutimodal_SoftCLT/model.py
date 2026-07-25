import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.NTXent_loss import *
from models.loss import loss_ntxent, loss_ntxent_class

import torch.optim as optim
# from tslearn.metrics import soft_dtw
from pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW
from itertools import islice

from tqdm import trange, tqdm
from scipy.stats import norm
from models.timelags import *
from models.soft_losses import *
from models.hard_losses import *

# class base_Model(nn.Module): # encoder
#     def __init__(self, input_channels=3, final_out_channels = 128, kernel_size = 25, stride = 3, features_len = 127):
#         super(base_Model, self).__init__()

#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
#                       stride=stride, bias=False, padding=(kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(0.35)
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(final_out_channels),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         model_output_dim = features_len
#         # self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
#         # self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

#         # self.lambda_ = args.lambda_ # 0.5
#         # self.soft_temporal = args.soft_temporal #True
#         # self.soft_instance = args.soft_instance #True

#         self.lambda_ = 0.5
#         self.soft_temporal = True
#         self.soft_instance = True
        
#         # self.tau_temp = args.tau_temp
#         self.tau_temp = 1

#     def softdtw_pairwise(self, x, sdtw):
#         """
#         x: (B, T, C) CUDA tensor
#         returns: (B, B)
#         """
#         B, T, C = x.shape

#         x_i = x.unsqueeze(1).expand(B, B, T, C)
#         x_j = x.unsqueeze(0).expand(B, B, T, C)

#         x_i = x_i.reshape(B * B, T, C)
#         x_j = x_j.reshape(B * B, T, C)

#         dist = sdtw(x_i, x_j)   # (B*B,)
#         return dist.view(B, B)
    
#     def softdtw_pairwise_blocked(self, x, sdtw, block=16):
#         B, T, C = x.shape
#         D = torch.zeros(B, B, device=x.device)

#         for i in range(0, B, block):
#             bi = min(block, B - i)
#             xi = x[i:i+bi]

#             for j in range(0, B, block):
#                 bj = min(block, B - j)
#                 xj = x[j:j+bj]

#                 xi_ = xi.unsqueeze(1).expand(bi, bj, T, C)
#                 xj_ = xj.unsqueeze(0).expand(bi, bj, T, C)

#                 xi_ = xi_.reshape(-1, T, C)
#                 xj_ = xj_.reshape(-1, T, C)

#                 D[i:i+bi, j:j+bj] = sdtw(xi_, xj_).view(bi, bj)

#                 # crucial for memory
#                 del xi_, xj_
#                 torch.cuda.empty_cache()

#         return D

#     def batch_soft_dtw_gpu(self, x , gamma=0.1):
#         """
#         x: (B, D, T) torch tensor on CUDA
#         returns: (B, B) distance matrix
#         """
#         N, D, T = x.shape
#         x = x.transpose(1, 2)  # (B, T, D)
        
#         # dist_mat = torch.zeros((N,N), device=x.device)
#         sdtw = SoftDTW(use_cuda=True, gamma=gamma)
#         mdtw_dist = self.softdtw_pairwise(x, sdtw)
#         # mdtw_dist = self.softdtw_pairwise_blocked(x, sdtw)
#         # print(mdtw_dist.shape)

#         dist_mat = mdtw_dist.fill_diagonal_(0)
#         dist_mat = dist_mat.to(x.device)
#         del mdtw_dist
#         # for i in range(N):
#         #     for j in range(i+1, N):
#         #         # mdtw_dist = soft_dtw(x[i], x[j], be="pytorch")
#         #         dist_mat[i,j] = mdtw_dist[i,j]
#         #         dist_mat[j,i] = mdtw_dist[i,j]
#         return dist_mat
    
    

#     def forward(self, aug1, aug2, train=False):
#         if train:
#             if self.soft_instance:
#                 soft_labels = self.batch_soft_dtw_gpu(aug1)
#                 # soft_labels = torch.tensor(soft_labels, device=aug1.device)
#                 soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
#                 del soft_labels
            
#             temporal_loss = torch.tensor(0., device=aug1.device)
#             instance_loss = torch.tensor(0., device=aug1.device)
            
#             #-------------------------------------------------#
#             # DEPTH = 1
#             #-------------------------------------------------#
#             aug1 = self.conv_block1(aug1)
#             aug2 = self.conv_block1(aug2)
#             # aug1 = aug1.transpose(1,2)
#             # aug2 = aug2.transpose(1,2)
#             # print(aug1.shape)
#             if self.soft_temporal:
#                 d=0
#                 timelag = timelag_sigmoid(aug1.shape[2], self.tau_temp*(2**d))
#                 timelag = torch.tensor(timelag, device=aug1.device)
#                 timelag_L, timelag_R = dup_matrix(timelag)
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
#             else:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
#             if self.soft_instance:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
#             else:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)

#             #-------------------------------------------------#
#             # DEPTH = 2
#             #-------------------------------------------------#
#             # aug1 = aug1.transpose(1,2)
#             # aug2 = aug2.transpose(1,2)
#             aug1 = self.conv_block2(aug1)
#             aug2 = self.conv_block2(aug2)
            

#             if self.soft_temporal:
#                 d=1
#                 timelag = timelag_sigmoid(aug1.shape[2],self.tau_temp*(2**d))
#                 timelag = torch.tensor(timelag, device=aug1.device)
#                 timelag_L, timelag_R = dup_matrix(timelag)
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
#             else:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
                
#             if self.soft_instance:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
#             else:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)
            
#             #-------------------------------------------------#
#             # DEPTH = 3
#             #-------------------------------------------------#
#             # aug1 = aug1.transpose(1,2)
#             # aug2 = aug2.transpose(1,2)
#             aug1 = self.conv_block3(aug1)
#             aug2 = self.conv_block3(aug2)
#             # aug1 = aug1.transpose(1,2)
#             # aug2 = aug2.transpose(1,2)
        
#             if self.soft_temporal:
#                 d=2
#                 timelag = timelag_sigmoid(aug1.shape[2],self.tau_temp*(2**d))
#                 timelag = torch.tensor(timelag, device=aug1.device)
#                 timelag_L, timelag_R = dup_matrix(timelag)

#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
#             else:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
           
#             if self.soft_instance:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
#                 del soft_labels_L, soft_labels_R
#             else:
#                 aug1 = aug1.transpose(1,2)
#                 aug2 = aug2.transpose(1,2)
#                 instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)
            
#             # feat1 = aug1
#             # feat2 = aug2

#             final_loss = temporal_loss + instance_loss

#             return aug1, aug2, final_loss

        
#         else:
#             feat1 = self.conv_block1(aug1)
#             feat1 = self.conv_block2(feat1)
#             feat1 = self.conv_block3(feat1)

#             # feat2 = self.conv_block1(aug2)
#             # feat2 = self.conv_block2(feat2)
#             # feat2 = self.conv_block3(feat2)
           
#             return feat1


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
        self.lambda_ = 0.5
        self.soft_temporal = True
        self.soft_instance = True
        
        self.tau_temp = 1
    
    

    def forward(self, aug1, aug2, soft_labels, train=False):
        if train:
            if self.soft_instance:
                soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
                del soft_labels
            
            temporal_loss = torch.tensor(0., device=aug1.device)
            instance_loss = torch.tensor(0., device=aug1.device)
            
            #-------------------------------------------------#
            # DEPTH = 1
            #-------------------------------------------------#
            aug1 = self.conv_block1(aug1)
            aug2 = self.conv_block1(aug2)
            if self.soft_temporal:
                d=0
                timelag = timelag_sigmoid(aug1.shape[2], self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=aug1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
            else:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
            if self.soft_instance:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
            else:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)

            #-------------------------------------------------#
            # DEPTH = 2
            #-------------------------------------------------#
            aug1 = self.conv_block2(aug1)
            aug2 = self.conv_block2(aug2)

            if self.soft_temporal:
                d=1
                timelag = timelag_sigmoid(aug1.shape[2],self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=aug1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
            else:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
                
            if self.soft_instance:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
            else:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)
            
            #-------------------------------------------------#
            # DEPTH = 3
            #-------------------------------------------------#
            aug1 = self.conv_block3(aug1)
            aug2 = self.conv_block3(aug2)
        
            if self.soft_temporal:
                d=2
                timelag = timelag_sigmoid(aug1.shape[2],self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=aug1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
            else:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
           
            if self.soft_instance:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
                del soft_labels_L, soft_labels_R
            else:
                aug1 = aug1.transpose(1,2)
                aug2 = aug2.transpose(1,2)
                instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)
        
        else:
            aug = self.conv_block1(aug1)
            aug = self.conv_block2(aug)
            aug = self.conv_block3(aug)
        
        ############################################################################
        ############################################################################
        
        if train:
            final_loss = temporal_loss + instance_loss
            return aug1, aug2, final_loss

        else:
            return aug





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
        num_iter=600,
        cut_off = 1,
        attract_filter = 0.8,
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

        def make_shared_projection_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, final_out_channels // 2),
                nn.BatchNorm1d(final_out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(final_out_channels // 2, final_out_channels // 4),
            )

        self.shared_projection_head_1 = make_shared_projection_head()
        self.shared_projection_head_2 = make_shared_projection_head()

        self.optimizer = optim.AdamW([{'params':self.EEG_encoder.parameters()},
                                        {'params':self.EOG_encoder.parameters()},
                                        {'params':self.EEG_contrasting.parameters()},
                                        {'params':self.EOG_contrasting.parameters()},
                                        {'params':self.shared_projection_head_1.parameters()},
                                        {'params':self.shared_projection_head_2.parameters()}], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
    
        self.option = option
        self.removal_option, self.position_option = option.split('_')[0], option.split('_')[1]
        # print(self.removal_option, self.position_option)

        self.topk = topk
        self.fn_mode = fn_mode
        # print('self.fn_mode:', self.fn_mode)

        self.use_iteration = use_iteration
        self.num_iter = num_iter

        self.aux_classifier = MLP(final_out_channels // 4, final_out_channels // 8, 5)

        if isinstance(cut_off, list):
            self.cutoff_uni, self.cutoff_multi = cut_off
            self.attract_filter_uni, self.attract_filter_multi = attract_filter
        else:
            self.cutoff_uni = cut_off
            self.cutoff_multi = cut_off
            self.attract_filter_uni = attract_filter
            self.attract_filter_multi = attract_filter
        # # self.cut_off = 1 #2 # 0.5 # 1
        # self.cut_off = cut_off #2 # 0.5 # 1
        # self.attract_filter = attract_filter
        # print("cut_off:", self.cut_off, "strategy_bound:", self.strategy_bound)
        
    
    def negative_cutoff(self, memory_module, neg_sim, cut_off): # redefine strategy threshold: use neg_sim
        instance_mean = {k: v[-1] \
                            for k, v in sorted(memory_module.items(), key=lambda item: item[1])}

        mu = np.mean(list(instance_mean.values()))
        sd = np.std(list(instance_mean.values()))

        #* global statistic
        # gaussian_norm = norm(mu, sd)

        #* thresholds for noisy negative pairs
        cutoff_bound = mu-cut_off*sd


        # for neg_sim based strategy threshold
        mu_neg = np.mean(neg_sim)
        sd_neg = np.std(neg_sim)

        #* global statistic
        # gaussian_norm = norm(mu, sd)


        strategy_threshold = mu_neg - cut_off * sd_neg

        return cutoff_bound, strategy_threshold


    def forward(self, train_loader, soft_labels, num_iter=600, cut_np=2, cut_fp=3):

        # torch.autograd.set_detect_anomaly(True)
        
        #* indicies of instance in dataset
        indicies = np.arange(len(train_loader.dataset))
        #* dictionary to record instance training loss (the memory module)
        # if self.removal_option in ['fn'] and self.position_option in ['A', 'AB', 'AC', 'ABC']:
        #     memory_module_T = dict([(k,[]) for k in indicies])
        #     memory_module_F = dict([(k,[]) for k in indicies])
        # if self.removal_option in ['fn'] and self.position_option in ['B', 'AB', 'BC', 'ABC']:
        #     memory_module_TF = dict([(k,[]) for k in indicies])

        neg_sim_all_T = []
        neg_sim_all_F = []
        neg_sim_all_TF = []

        losses = []
        # torch.autograd.set_detect_anomaly(True)

        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4
        lambda_4 = 0.1
        
        pbar = trange(self.num_epochs)
        for e in pbar:
            
            if not self.use_iteration:
                num_cancel_t_total, num_attract_t_total = 0, 0
                num_cancel_f_total, num_attract_f_total = 0, 0
                num_cancel_tf_total, num_attract_tf_total = 0, 0
                # num_b = 0
                # print(len(train_loader))
                # for batch in tqdm(islice(train_loader, num_iter)):
                for batch in train_loader:
                # for batch in train_loader:
                    EEG, EOG, EEG_aug, EOG_aug, index = batch
                    if index.shape[0] < 2:  # the case when there is only one sample in the batch
                        continue
                    # print("index", index)
                    # num_b += 1
                    # if num_b == num_iter+1:
                    #     break
                    EEG = EEG.to(self.device)
                    EOG = EOG.to(self.device)
                    EEG_aug = EEG_aug.to(self.device)
                    EOG_aug = EOG_aug.to(self.device)
                    x = (EEG, EOG, EEG_aug, EOG_aug)

                    # soft_labels_batch = soft_labels[index][:,index].to(self.device)

                    index_np = index.detach().cpu().numpy()

                    if isinstance(soft_labels, torch.Tensor):
                        sim_batch = soft_labels.detach().cpu()[index_np][:, index_np].numpy()
                    else:
                        sim_batch = np.asarray(soft_labels[np.ix_(index_np, index_np)], dtype=np.float32)

                    # Equivalent to densify(-(1 - sim_mat), 6, 0.5), but only for
                    # the current B x B batch instead of the full N x N matrix.
                    tau = 6.0
                    alpha = 0.5
                    soft_labels_np = (2 * alpha) / (1 + np.exp(-tau * (sim_batch - 1.0)))
                    soft_labels_np += (1 - alpha) * np.eye(len(index_np), dtype=np.float32)
                    soft_labels_batch = torch.from_numpy(soft_labels_np.astype(np.float32, copy=False)).to(self.device)
                    # if e >= self.warm_epochs and self.removal_option in ['fn']: # False negative cancellation
                    #     sum_1, L_C_T, L_C_F, L_C_TF, pos_sim_T, pos_sim_F, pos_sim_TF, neg_sim_T, neg_sim_F, neg_sim_TF, num_cancel_t, num_attract_t, num_cancel_f, num_attract_f, num_cancel_tf, num_attract_tf = self.loss_compute(x, pos_option = self.position_option, bounds = bounds)
                    #     num_cancel_t_total += num_cancel_t
                    #     num_attract_t_total += num_attract_t
                    #     num_cancel_f_total += num_cancel_f
                    #     num_attract_f_total += num_attract_f
                    #     num_cancel_tf_total += num_cancel_tf
                    #     num_attract_tf_total += num_attract_tf
                    
                    # else:
                    soft_loss, sum_1, L_C_T, L_C_F, L_C_TF, pos_sim_T, pos_sim_F, pos_sim_TF, neg_sim_T, neg_sim_F, neg_sim_TF, _, _, _, _, _, _ = self.loss_compute(x, soft_labels_batch, pos_option = 'None')
                    
                    # #* update the memory module
                    # if self.removal_option in ['fn'] and self.position_option in ['A', 'AB', 'AC', 'ABC']: # DBPM reweights after warmup
                    #     for i in range(len(index)):
                    #         memory_module_T[index[i].cpu().item()].append(pos_sim_T[i].cpu().item())
                    #         memory_module_F[index[i].cpu().item()].append(pos_sim_F[i].cpu().item())
                    # if self.removal_option in ['fn'] and self.position_option in ['B', 'AB', 'BC', 'ABC']:
                    #     for i in range(len(index)):
                    #         memory_module_TF[index[i].cpu().item()].append(pos_sim_TF[i].cpu().item())
             
                    # ########  Update Histogram  ########
                    # if hasattr(neg_sim_T, 'cpu'): 
                    #     neg_sim_T = neg_sim_T.detach().cpu().numpy()
                    # if hasattr(neg_sim_F, 'cpu'): 
                    #     neg_sim_F = neg_sim_F.detach().cpu().numpy()
                    # if hasattr(neg_sim_TF, 'cpu'): 
                    #     neg_sim_TF = neg_sim_TF.detach().cpu().numpy()

                    # neg_sim_all_T.extend(neg_sim_T.tolist())
                    # neg_sim_all_F.extend(neg_sim_F.tolist())
                    # neg_sim_all_TF.extend(neg_sim_TF.tolist())

                    loss_C_T = torch.mean(L_C_T)
                    loss_C_F = torch.mean(L_C_F)
                    loss_C_TF = torch.mean(L_C_TF)  

                    # if self.removal_option in ['fn'] and self.position_option in ['B', 'AB', 'BC', 'ABC']:
                    #     loss = lambda_1*sum_1 + lambda_2*(loss_C_T + loss_C_F) + lambda_3*loss_C_TF + lambda_4*aux_loss
                    # else:
                    loss = lambda_1*sum_1 + lambda_2*(loss_C_T + loss_C_F) + lambda_3*loss_C_TF

                    loss = soft_loss + 0.7*loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss))

                    

                # if e >= self.warm_epochs and self.removal_option in ['fn']:
                #     print("Epoch {}: num_cancel_t_total {}, num_attract_t_total {}, num_cancel_f_total {}, num_attract_f_total {}, num_cancel_tf_total {}, num_attract_tf_total {}".format(
                #         e, num_cancel_t_total, num_attract_t_total, num_cancel_f_total, num_attract_f_total, num_cancel_tf_total, num_attract_tf_total
                #     ))
                
            self.scheduler.step(e)
            pbar.set_description(str(sum(losses) / len(losses)))


    def loss_compute(self, x, soft_labels_batch, pos_option = 'None', bounds = None):
        # print('loss_compute fn_mode:', fn_mode)
        if bounds is not None:
            cutoff_bound_t, strategy_threshold_t, cutoff_bound_f, strategy_threshold_f, cutoff_bound_tf, strategy_threshold_tf = bounds

        EEG, EOG, EEG_aug, EOG_aug = x[0], x[1], x[2], x[3]
        # EEG_feat = self.EEG_encoder(EEG)
        # EEG_aug_feat = self.EEG_encoder(EEG_aug)
        # EOG_feat = self.EOG_encoder(EOG)
        # EOG_aug_feat = self.EOG_encoder(EOG_aug)

        EEG_feat, EEG_aug_feat, EEG_loss = self.EEG_encoder(EEG, EEG_aug, soft_labels_batch, train=True)
        EOG_feat, EOG_aug_feat, EOG_loss = self.EOG_encoder(EOG, EOG_aug, soft_labels_batch, train=True)


        EEG_feat = F.normalize(EEG_feat, dim=1)
        EEG_aug_feat = F.normalize(EEG_aug_feat, dim=1)
        EOG_feat = F.normalize(EOG_feat, dim=1)
        EOG_aug_feat = F.normalize(EOG_aug_feat, dim=1)

        # print("EEG_feat shape:", EEG_feat.shape)

        # if pos_option in ['C', 'AC', 'BC', 'ABC']:
        #     # print("Pos_option C, filtering")
        #     L_T, z_T, c_T = self.EEG_contrasting(EEG_feat, EEG_aug_feat, filter=True)
        #     L_T_aug, _, c_T_aug = self.EEG_contrasting(EEG_aug_feat, EEG_feat, filter=True)

        #     L_F, z_F, c_F = self.EOG_contrasting(EOG_feat, EOG_aug_feat, filter=True)
        #     L_F_aug, _, c_F_aug = self.EOG_contrasting(EOG_aug_feat, EOG_feat, filter=True)
        # else:
        L_T, z_T, c_T = self.EEG_contrasting(EEG_feat, EEG_aug_feat)
        L_T_aug, _, c_T_aug = self.EEG_contrasting(EEG_aug_feat, EEG_feat)

        L_F, z_F, c_F = self.EOG_contrasting(EOG_feat, EOG_aug_feat)
        L_F_aug, _, c_F_aug = self.EOG_contrasting(EOG_aug_feat, EOG_feat)

        pos_sim_T = None
        pos_sim_F = None
        num_cancel_t, num_attract_t = 0, 0
        num_cancel_f, num_attract_f = 0, 0
        # if pos_option in ['A', 'AB', 'AC', 'ABC']:
        #     # print("Applying false negative cancellation in position A...")
        #     # print("Pos_option A, filtering")
        #     L_C_T, pos_sim_T, neg_sim_T, num_cancel_t, num_attract_t = loss_ntxent([c_T, c_T_aug], self.device, filter=True, cutoff_bound = cutoff_bound_t, strategy_threshold = strategy_threshold_t, position='A', attract_filter = self.attract_filter_uni)
        #     L_C_F, pos_sim_F, neg_sim_F, num_cancel_f, num_attract_f = loss_ntxent([c_F, c_F_aug], self.device, filter=True, cutoff_bound = cutoff_bound_f, strategy_threshold = strategy_threshold_f, position='A', attract_filter = self.attract_filter_uni)
        # else:
        L_C_T, pos_sim_T, neg_sim_T, _, _ = loss_ntxent([c_T, c_T_aug], self.device, filter=False)
        L_C_F, pos_sim_F, neg_sim_F, _, _ = loss_ntxent([c_F, c_F_aug], self.device, filter=False)

        # print("L_C_T", L_C_T.shape)
        
        ############################ Compute Contrastive Loss Instead ##############################
        pos_sim_TF = None
        num_cancel_tf, num_attract_tf = 0, 0

        classes_T = self.aux_classifier(c_T.reshape(c_T.shape[0], -1))
        classes_F = self.aux_classifier(c_F.reshape(c_F.shape[0], -1))
        classes_T_num = torch.argmax(classes_T, dim=1)
        classes_F_num = torch.argmax(classes_F, dim=1)

        c_T_shared = self.shared_projection_head_1(z_T)
        c_F_shared = self.shared_projection_head_2(z_F)

        # if pos_option in ['B', 'AB', 'BC', 'ABC']:
        #     # print("Pos_option B, filtering")
        #     # L_C_TF_1, pos_sim_TF_1, neg_sim_TF_1, num_cancel_tf_1, num_attract_tf_1 = loss_ntxent([c_T, c_F], self.device, filter=True, cutoff_bound = cutoff_bound_tf, strategy_threshold = strategy_threshold_tf)
        #     # L_C_TF_2, pos_sim_TF_2, neg_sim_TF_2, num_cancel_tf_2, num_attract_tf_2 = loss_ntxent([c_F, c_T], self.device, filter=True, cutoff_bound = cutoff_bound_tf, strategy_threshold = strategy_threshold_tf)
            
        #     L_C_TF_1, pos_sim_TF_1, neg_sim_TF_1, num_cancel_tf_1, num_attract_tf_1 = loss_ntxent([c_T_shared, c_F_shared], self.device, filter=True, cutoff_bound = cutoff_bound_tf, strategy_threshold = strategy_threshold_tf, position='B', attract_filter = self.attract_filter_multi)
        #     L_C_TF_2, pos_sim_TF_2, neg_sim_TF_2, num_cancel_tf_2, num_attract_tf_2 = loss_ntxent([c_F_shared, c_T_shared], self.device, filter=True, cutoff_bound = cutoff_bound_tf, strategy_threshold = strategy_threshold_tf, position='B', attract_filter = self.attract_filter_multi)
            
        #     L_C_TF = (L_C_TF_1 + L_C_TF_2) / 2
        #     pos_sim_TF = torch.cat((pos_sim_TF_1, pos_sim_TF_2), dim=0)
        #     neg_sim_TF = torch.cat((neg_sim_TF_1, neg_sim_TF_2), dim=0)
        #     num_cancel_tf = num_cancel_tf_1 + num_cancel_tf_2
        #     num_attract_tf = num_attract_tf_1 + num_attract_tf_2
        # else:
            # L_C_TF_1, pos_sim_TF_1, neg_sim_TF_1, _, _ = loss_ntxent([c_T, c_F], self.device, filter=False)
            # L_C_TF_2, pos_sim_TF_2, neg_sim_TF_2, _, _ = loss_ntxent([c_F, c_T], self.device, filter=False)
            
        L_C_TF_1, pos_sim_TF_1, neg_sim_TF_1, _, _ = loss_ntxent([c_T_shared, c_F_shared], self.device, filter=False)
        L_C_TF_2, pos_sim_TF_2, neg_sim_TF_2, _, _ = loss_ntxent([c_F_shared, c_T_shared], self.device, filter=False)
        
        L_C_TF = (L_C_TF_1 + L_C_TF_2) / 2
        pos_sim_TF = torch.cat((pos_sim_TF_1, pos_sim_TF_2), dim=0)
        neg_sim_TF = torch.cat((neg_sim_TF_1, neg_sim_TF_2), dim=0)
        
        ############################ Compute Contrastive Loss End ##############################
        soft_loss = EEG_loss + EOG_loss
        
        sum_1 = L_T+L_T_aug+L_F+L_F_aug

        return soft_loss, sum_1, L_C_T, L_C_F, L_C_TF, pos_sim_T, pos_sim_F, pos_sim_TF, neg_sim_T, neg_sim_F, neg_sim_TF, num_cancel_t, num_attract_t, num_cancel_f, num_attract_f, num_cancel_tf, num_attract_tf

        



class Model(nn.Module):
    def __init__(self, 
        soft_labels,
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
        option = 'bp_A',
        topk = 0.25,
        fn_mode = 'cancel',
        use_iteration=False,
        num_iter=600,
        cut_off = 1,
        attract_filter = 0.8,
        small_classifier = False,
        depth=4,
        ):
        super(Model, self).__init__()
        
        # self.EEG_dims = EEG_dims
        # self.EOG_dims = EOG_dims

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
        option = option,
        topk = topk,
        fn_mode = fn_mode,
        use_iteration=use_iteration,
        num_iter=num_iter,
        cut_off = cut_off,
        attract_filter = attract_filter,
        depth=depth
        )
        self.classifier = MLP(hidden_dim * 2, hidden_dim, num_class)
        self.soft_labels = soft_labels

    def forward(self, x, ssl = False, num_iter = 200):
        if ssl == False: # input will be a single sample
            EEG, EOG = x[0], x[1]
            EEG_features = self.EEG_encoder(EEG, EEG, soft_labels=None, train=False)
            EOG_features = self.EOG_encoder(EOG, EOG, soft_labels=None, train=False)
            EEG_transformer_features = self.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2)).reshape(EEG.shape[0], -1)
            EOG_transformer_features = self.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2)).reshape(EOG.shape[0], -1)
            features_flat = torch.cat((EEG_transformer_features, EOG_transformer_features), dim=1)
            # print("features", features.shape)
            # features_flat = features.reshape(features.shape[0], -1)
            return self.classifier(features_flat)
        return self.TFCC(x, self.soft_labels, num_iter=num_iter) #input will be a dataloader
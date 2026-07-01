
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class InfoNCE(nn.Module):
    def __init__(self, temperature, device, filter=False, cutoff_bound = -1, strategy_threshold = -1, position=None, attract_filter=0.8):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.filter = filter
        self.cutoff_bound = cutoff_bound 
        self.strategy_threshold = strategy_threshold
        self.position = position
        self.attract_filter = attract_filter


    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, out_1, out_2):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        bsz = out_1.shape[0]

        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        out_1_dist = out_1
        out_2_dist = out_2

        # out = torch.cat([out_1, out_2], dim=0)
        # out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        pos_sim = torch.sum(out_1 * out_2, dim=-1)
        pos = torch.exp(pos_sim / self.temperature)

        # pos = torch.cat([pos, pos], dim=0)
        # print('pos:', pos.shape)

        # similarity = torch.exp(torch.mm(out, out_dist.t()) / self.temperature)
        neg_sim = torch.mm(out_1, out_2.t())
        # similarity = torch.exp(torch.mm(out_1, out_2.t()) / self.temperature)

        # neg_mask = self.mask_correlated_samples(bsz).to(self.device)
        neg_mask = torch.ones((bsz, bsz)).to(self.device)
        neg_mask = neg_mask.fill_diagonal_(0)

        neg_temp = neg_sim * neg_mask
        
        num_cancel, num_attract = 0, 0
        if self.filter:
            if self.cutoff_bound != -1 and self.strategy_threshold != -1:
                # cancel_mask = torch.where((neg_temp < self.strategy_threshold) & (neg_temp > self.cutoff_bound), 1.0, 0.0).bool() # filter false negatives for cancellation
                # attract_mask = torch.where(neg_temp > self.strategy_threshold, 1.0, 0.0).bool() # filter false negatives for attraction
                
                pos_temp = pos_sim.unsqueeze(1)

                # now, lose the strategy_threshold
                cancel_mask = torch.where((neg_temp > self.cutoff_bound), 1.0, 0.0).bool() # filter false negatives for cancellation
                
                neg_pos_sim = torch.mm(out_2, out_2.t())
                attract_neg_mask = (neg_pos_sim > self.attract_filter)
                # # only changed to the following line
                # attract_neg_mask = (torch.abs(neg_pos_sim - pos_temp) < 0.1)
                
                attract_mask = torch.where(attract_neg_mask & cancel_mask, 1.0, 0.0).bool() # filter false negatives for attraction

                
                # assert torch.all(pos_temp[:, 0] == pos_temp[:, 1]), "pos_temp should have equal columns"
                cancel_weights = torch.abs((neg_temp -  pos_temp) * cancel_mask)
                # softmax normalization
                # cancel_weights = F.softmax(cancel_weights, dim=1)
                
                sum_total = torch.sum(torch.exp(cancel_weights) * cancel_mask, dim=1, keepdims=True) + 1e-8  # avoid division by 0
                cancel_weight = torch.exp(cancel_weights) * cancel_mask / sum_total
                cancel_weights = torch.where(cancel_mask, cancel_weight, 1.0)
                
                if self.position == 'A':
                    attract_weights = torch.abs((neg_temp -  pos_temp) * attract_mask)
                    # attract_weights = torch.ones(neg_temp.shape).to(self.device) - F.softmax(attract_weights, dim=1)
                    attract_weights = torch.ones(neg_temp.shape).to(self.device) - attract_weights

                    # only add this line to normalize the attract weights
                    attract_weights = attract_weights / (attract_weights.sum(dim=1, keepdim=True) + 1e-8)

                   
                elif self.position == 'B':  
                    # add the following line to set all attract weights to 1
                    attract_weights = attract_mask.float()

                # for negative, remove samples for attraction and perform weighted cancellation
                similarity = torch.exp(neg_temp / self.temperature)
                similarity_neg = similarity * ~attract_mask
                similarity_neg = similarity_neg * cancel_weights


                if self.position == 'B' or self.position == 'A':  # don't do attraction for fn_A
                    # add the following line to avoid collapse
                    alpha = 0.5
                    new_pos = alpha * torch.sum(similarity * attract_weights, dim=1)
                    
                    # for positivel, perform weighted attraction
                    # new_pos = torch.sum(similarity * attract_weights, dim=1)
                    pos = pos + new_pos


                neg = torch.sum(similarity_neg, 1)
                logits = -torch.log(pos / (pos + neg))
                
                neg_sim_return = neg_sim[(neg_mask.bool() & ~attract_mask.bool())]  # when computing negative statistics lose the samples for attraction

                # check how many samples are being weights
                num_cancel = torch.sum(cancel_mask).item()
                num_attract = torch.sum(attract_mask).item()

        else:
            similarity_neg = torch.exp(neg_temp / self.temperature)
            neg = torch.sum(similarity_neg, 1)
            logits = -torch.log(pos / (pos + neg))

            neg_sim_return = neg_sim[neg_mask.bool()]  # when computing negative statistics lose the samples for attraction

        # logits_re = logits.view(2, -1)
        logits_re = logits
        pos_loss = torch.mean(logits_re, dim=0)


        
        return pos_loss, pos_sim, neg_sim_return, num_cancel, num_attract

def loss_ntxent(features, device, filter=False, cutoff_bound = -1, strategy_threshold = -1, position='None', attract_filter=0.8):
    fn = InfoNCE(temperature=0.2, device=device, filter=filter, cutoff_bound = cutoff_bound, strategy_threshold = strategy_threshold, position=position, attract_filter=attract_filter) # topk: the percentage of each batch to be filtered
    pos_loss, pos_sim, neg_sim, num_cancel, num_attract = fn(features[0], features[1])
    return pos_loss, pos_sim, neg_sim, num_cancel, num_attract










class InfoNCE_class(nn.Module):
    def __init__(self, temperature, device, filter=False, cutoff_bound = -1, strategy_threshold = -1):
        super(InfoNCE_class, self).__init__()
        self.device = device
        self.temperature = temperature
        self.filter = filter
        self.cutoff_bound = cutoff_bound 
        self.strategy_threshold = strategy_threshold


    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, out_1, out_2, classes_1, classes_2):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        bsz = out_1.shape[0]

        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        out_1_dist = out_1
        out_2_dist = out_2

        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        pos_sim = torch.sum(out_1 * out_2, dim=-1)
        pos = torch.exp(pos_sim / self.temperature)

        # pos = torch.cat([pos, pos], dim=0)
        # print('pos:', pos.shape)

        neg_sim = torch.mm(out_1, out_2.t())

        # neg_mask = self.mask_correlated_samples(bsz).to(self.device)
        neg_mask = torch.ones((bsz, bsz)).to(self.device)
        neg_mask = neg_mask.fill_diagonal_(0)

        neg_temp = neg_sim * neg_mask
        
        num_cancel, num_attract = 0, 0
        # if self.filter: # the class only used for filtering, no need to do the conditioning here
        if self.cutoff_bound != -1 and self.strategy_threshold != -1:
            cancel_mask = torch.where((neg_temp < self.strategy_threshold) & (neg_temp > self.cutoff_bound), 1.0, 0.0).bool() # filter false negatives for cancellation
            attract_mask = torch.where(neg_temp > self.strategy_threshold, 1.0, 0.0).bool() # filter false negatives for attraction

            class_mask = classes_1.unsqueeze(1).eq(classes_2.unsqueeze(0))
            attract_mask &= class_mask
            cancel_mask  &= class_mask

            pos_temp = pos_sim.unsqueeze(1)
            # assert torch.all(pos_temp[:, 0] == pos_temp[:, 1]), "pos_temp should have equal columns"
            cancel_weights = torch.abs((neg_temp -  pos_temp) * cancel_mask)
            attract_weights = torch.abs((neg_temp -  pos_temp) * attract_mask)
            # softmax normalization
            cancel_weights = F.softmax(cancel_weights, dim=1)
            ### keeping the true negative samples!
            zero_mask = (cancel_weights == 0.0)
            cancel_weights = torch.where(zero_mask, torch.ones_like(cancel_weights), cancel_weights)
            

            attract_weights = torch.ones(neg_temp.shape).to(self.device) - F.softmax(attract_weights, dim=1)

            # for negative, remove samples for attraction and perform weighted cancellation
            similarity = torch.exp(neg_temp / self.temperature)
            similarity_neg = similarity * ~attract_mask
            similarity_neg = similarity_neg * cancel_weights

            # for positivel, perform weighted attraction
            new_pos = torch.sum(similarity * attract_weights, dim=1)
            pos = pos + new_pos

            # check how many samples are being weights
            num_cancel = torch.sum(cancel_mask).item()
            num_attract = torch.sum(attract_mask).item()

            # print("pos_B", pos)
            # print("new_pos_B", new_pos)
            # print("pos_B value", torch.sum(pos).item())
            # print("Neg_B", similarity)
            # print("Neg_B value", torch.sum(similarity).item())
        
        else:
            similarity_neg = torch.exp(neg_temp / self.temperature)


        neg = torch.sum(similarity_neg, 1)
        # print("neg", neg)
        # print("neg shape", neg.shape)
        # print("neg value", torch.sum(neg).item())

        # logits = -torch.log(pos / (pos + neg))
        # epsilon = 1e-8 # A tiny constant
        logits = -torch.log(pos / (pos + neg))
        # logits_re = logits.view(2, -1)
        logits_re = logits
        pos_loss = torch.mean(logits_re, dim=0)

        return pos_loss, pos_sim, num_cancel, num_attract

def loss_ntxent_class(features, classes, device, filter=False, cutoff_bound = -1, strategy_threshold = -1):
    fn = InfoNCE_class(temperature=0.2, device=device, filter=filter, cutoff_bound = cutoff_bound, strategy_threshold = strategy_threshold) # topk: the percentage of each batch to be filtered
    pos_loss, pos_sim, num_cancel, num_attract = fn(features[0], features[1], classes[0], classes[1])
    return pos_loss, pos_sim, num_cancel, num_attract
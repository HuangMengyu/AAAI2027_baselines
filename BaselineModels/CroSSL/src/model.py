import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.TC import TC
# from models.NTXent_loss import *
# from models.loss import loss_ntxent, loss_ntxent_class
import model_utils.loss as loss

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

# ------------------------------------------------------------------------- #
# class CMContrastiveModel(tf.keras.Model):
#     def __init__(self, embedding_model, projection_model, coverage, masking, masking_threshold=0.9, **kwargs):
#         super().__init__()
#         self.encoder = embedding_model
#         self.projector = projection_model
#         self.coverage_rate = coverage
#         self._similarity_layer = DotProduct()
#         self.masking_strategy = masking
#         self.mask_threshold = masking_threshold

#     def get_random_indices(self, data_shape):
#         # Create random indices from a uniform distribution and then split
#         # it into mask and unmask indices.
#         if self.masking_strategy == 'random':
#             rand = tf.random.uniform(shape=data_shape)
#             mask_a = tf.where((rand < self.mask_threshold), 1.0, 0.0)
#             rand = tf.random.uniform(shape=data_shape)
#             mask_b = tf.where((rand < self.mask_threshold), 1.0, 0.0)

#         elif self.masking_strategy == 'spatial':
#             # indices = tf.argsort(tf.random.uniform(shape=(data_shape[0], data_shape[1])), axis=1)
#             # indices_a = tf.expand_dims(indices[:, :self.coverage_rate], axis=-1)
#             # indices_b = tf.expand_dims(indices[:, -self.coverage_rate:], axis=-1)
#             indices = tf.random.uniform(shape=(data_shape[0], data_shape[1]))
#             indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
#             mask_a = tf.expand_dims(indices, axis=-1)
#             mask_a = tf.repeat(mask_a, repeats=data_shape[-1], axis=-1)

#             indices = tf.random.uniform(shape=(data_shape[0], data_shape[1]))
#             indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
#             mask_b = tf.expand_dims(indices, axis=-1)
#             mask_b = tf.repeat(mask_b, repeats=data_shape[-1], axis=-1)

#         elif self.masking_strategy == 'temporal':
#             # indices = tf.argsort(tf.random.uniform(shape=(data_shape[0], data_shape[1])), axis=1)
#             # indices_a = tf.expand_dims(indices[:, :self.coverage_rate], axis=-1)
#             # indices_b = tf.expand_dims(indices[:, -self.coverage_rate:], axis=-1)
#             indices = tf.random.uniform(shape=(data_shape[0], data_shape[2]))
#             indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
#             mask_a = tf.expand_dims(indices, axis=1)
#             mask_a = tf.repeat(mask_a, repeats=data_shape[-1], axis=-1)

#             indices = tf.random.uniform(shape=(data_shape[0], data_shape[1]))
#             indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
#             mask_b = tf.expand_dims(indices, axis=-1)
#             mask_b = tf.repeat(mask_b, repeats=data_shape[-1], axis=-1)
#         return mask_a, mask_b

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             modality_embeddings = self.encoder(data, training=False)

#             modality_embeddings = tf.stack(modality_embeddings)
#             # dim_size = len(modality_embeddings)
#             modality_embeddings = tf.transpose(modality_embeddings, (1, 0, 2))  # Batch x Mod x code_size
#             ind_a, ind_b = self.get_random_indices(modality_embeddings.shape)

#             # if self.masking_strategy == 'random':
#             #    comb_emb_a = modality_embeddings * ind_a
#             #    comb_emb_b = modality_embeddings * ind_b
#             # elif self.masking_strategy == 'spatial':
#             #    comb_emb_a = tf.gather_nd(batch_dims=1, indices=ind_a, params=modality_embeddings)
#             #    comb_emb_b = tf.gather_nd(batch_dims=1, indices=ind_b, params=modality_embeddings)

#             comb_emb_a = modality_embeddings * ind_a
#             comb_emb_b = modality_embeddings * ind_b

#             rep_a, rep_b = self.projector(comb_emb_a), self.projector(comb_emb_b)

#             loss = self.compiled_loss(rep_a, rep_b)
#             loss += sum(self.losses)
#             # loss += sum(self.projector.losses)

#         trainable_vars = self.encoder.trainable_variables + self.projector.trainable_variables
#         gradients = tape.gradient(loss, self.encoder.trainable_variables + self.projector.trainable_variables, )
#         self.optimizer.apply_gradients(
#             zip(gradients, self.encoder.trainable_variables + self.projector.trainable_variables))
#         return {m.name: m.result() for m in self.metrics}

import torch
import torch.nn as nn
import torch.nn.functional as F

class CMContrastiveModel(nn.Module):
    def __init__(self, embedding_model, projection_model, compiled_loss, coverage, masking, masking_threshold=0.9):
        super(CMContrastiveModel, self).__init__()
        self.encoder = embedding_model
        self.projector = projection_model
        self.coverage_rate = coverage
        # self._similarity_layer = DotProduct()
        self.masking_strategy = masking
        self.mask_threshold = masking_threshold
        self.compiled_loss = compiled_loss

        self.gap = nn.AdaptiveAvgPool1d(1)

    def get_random_indices(self, data_shape, device):
        # TensorFlow uses (Batch, Modality, Code_Size)
        # data_shape: [batch_size, num_modalities, embedding_dim]
        
        mask_a, mask_b = None, None
        
        if self.masking_strategy == 'random':
            rand_a = torch.rand(data_shape, device=device)
            mask_a = torch.where(rand_a < self.mask_threshold, 1.0, 0.0)
            
            rand_b = torch.rand(data_shape, device=device)
            mask_b = torch.where(rand_b < self.mask_threshold, 1.0, 0.0)

        elif self.masking_strategy == 'spatial':
            # Masking across the modality dimension (dim 1)
            indices_a = torch.rand((data_shape[0], data_shape[1]), device=device)
            indices_a = torch.where(indices_a < self.mask_threshold, 1.0, 0.0)
            mask_a = indices_a.unsqueeze(-1).repeat(1, 1, data_shape[-1])

            indices_b = torch.rand((data_shape[0], data_shape[1]), device=device)
            indices_b = torch.where(indices_b < self.mask_threshold, 1.0, 0.0)
            mask_b = indices_b.unsqueeze(-1).repeat(1, 1, data_shape[-1])

        elif self.masking_strategy == 'temporal':
            # Masking across the embedding/time dimension (dim 2)
            indices_a = torch.rand((data_shape[0], data_shape[2]), device=device)
            indices_a = torch.where(indices_a < self.mask_threshold, 1.0, 0.0)
            mask_a = indices_a.unsqueeze(1).repeat(1, data_shape[1], 1)

            indices_b = torch.rand((data_shape[0], data_shape[1]), device=device)
            indices_b = torch.where(indices_b < self.mask_threshold, 1.0, 0.0)
            mask_b = indices_b.unsqueeze(-1).repeat(1, 1, data_shape[-1])

        return mask_a, mask_b

    def forward(self, data):
        """
        Forward pass roughly equivalent to one 'train_step' logic.
        """
        # 1. Get modality embeddings from encoder
        # PyTorch encoders usually return a single tensor or list of tensors
        
        assert isinstance(self.encoder, list) and isinstance(data, tuple), "Encoder should be a list of models and data should be a tuple of inputs."
        
        for i, enc in enumerate(self.encoder):
            if i == 0:
                modality_embeddings = enc(data[i])
                modality_embeddings = self.gap(modality_embeddings).squeeze(-1)
                modality_embeddings = [modality_embeddings]
            else:
                modality_embeddings += [self.gap(enc(data[i])).squeeze(-1)]
                # modality_embeddings += [enc(data[i])]
        
        if isinstance(modality_embeddings, list):
            modality_embeddings = torch.stack(modality_embeddings) # [Mod, Batch, Dim]
            # assert modality_embeddings.shape == (2, 128, 128), modality_embeddings.shape
            modality_embeddings = modality_embeddings.permute(1, 0, 2) # [Batch, Mod, Dim]

        # 2. Generate Masks
        ind_a, ind_b = self.get_random_indices(modality_embeddings.shape, modality_embeddings.device)

        # 3. Apply Masks
        comb_emb_a = modality_embeddings * ind_a
        comb_emb_b = modality_embeddings * ind_b

        # 4. Project
        rep_a = self.projector(comb_emb_a)
        rep_b = self.projector(comb_emb_b)

        loss = self.compiled_loss(rep_a, rep_b)
        

        return loss

class CroSSL(nn.Module):
    def __init__(self, device,
        EEG_encoder,
        EOG_encoder,
        projector,
        loss_calculator,
        batch_size=32,
        num_epochs=100,
        lr = 1e-4,
        ):
        super(CroSSL, self).__init__()
        self.EEG_encoder = EEG_encoder
        self.EOG_encoder = EOG_encoder  
        self.projector = projector
        self.loss_calculator = loss_calculator 
        
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        # self.warm_epochs = warm_epochs
        self.device = device
        


        self.optimizer = optim.AdamW([{'params':self.EEG_encoder.parameters()},
                                        {'params':self.EOG_encoder.parameters()},
                                        {'params':self.projector.parameters()},
                                        ], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
    
    def forward(self, train_loader, cut_np=2, cut_fp=3):
        losses = []
        
        pbar = trange(self.num_epochs)
        for e in pbar:  
            for batch in train_loader:
                EEG, EOG = batch
                if EEG.shape[0] < 2:  # the case when there is only one sample in the batch
                    continue
                # print("index", index)
                EEG = EEG.to(self.device)
                EOG = EOG.to(self.device)
                x = (EEG, EOG)
                # EEG_aug = EEG_aug.to(self.device)
                # EOG_aug = EOG_aug.to(self.device)
                # x = (EEG, EOG, EEG_aug, EOG_aug)

                loss = self.loss_calculator(x)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))      
            
            self.scheduler.step(e)
            pbar.set_description(str(sum(losses) / len(losses)))


        



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
        batch_size=32,
        num_epochs=100,
        lr = 1e-4,
        ):
        super(Model, self).__init__()
        
        # self.EEG_dims = EEG_dims
        # self.EOG_dims = EOG_dims

        self.EEG_encoder = base_Model(mod1_dims, final_out_channels, kernel_size, stride, output_dims)
        self.EOG_encoder = base_Model(mod2_dims, final_out_channels, kernel_size, stride, output_dims)
        

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.BatchNorm1d(proj_size), # Optional: PyTorch uses 1D for flat features
            nn.Linear(32, 32),
            # nn.BatchNorm1d(proj_size),
            nn.LayerNorm(32)
        )

        custom_loss_obj = loss.CustomLoss(temperature=0.2, sim_coeff=10,
                                      std_coeff=10, cov_coeff=100)
        loss_fn = custom_loss_obj.get_loss_fn('vicreg')

        self.loss_calculator = CMContrastiveModel([self.EEG_encoder, self.EOG_encoder], self.projector, loss_fn, 2, 'spatial', masking_threshold=0.9)

        self.CroSSL = CroSSL(device,
        EEG_encoder = self.EEG_encoder,
        EOG_encoder = self.EOG_encoder, 
        projector = self.projector,
        loss_calculator = self.loss_calculator,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr = lr
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = MLP(128 * 2, 128, num_class)
        # self.classifier = MLP(128 * 2, 128, num_class)

    def forward(self, x, ssl = False):
        if ssl == False: # input will be a single sample
            EEG, EOG = x[0], x[1]
            EEG_features = self.EEG_encoder(EEG)
            EEG_features = self.gap(EEG_features)
            # EEG_features = self.projector(EEG_features)
            EOG_features = self.EOG_encoder(EOG)
            EOG_features = self.gap(EOG_features)
            # EOG_features = self.projector(EOG_features)

            EEG_features_flat = EEG_features.reshape(EEG.shape[0], -1)
            EOG_features_flat = EOG_features.reshape(EOG.shape[0], -1)
       
            features_flat = torch.cat((EEG_features_flat, EOG_features_flat), dim=1)
       
            return self.classifier(features_flat)
        return self.CroSSL(x) #input will be a dataloader
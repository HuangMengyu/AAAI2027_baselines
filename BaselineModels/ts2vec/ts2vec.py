import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.encoder import Transformer, cnn_extractor, Seq_Transformer
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math

class TS2Vec(nn.Module):
    '''The TS2Vec model'''
    
    def __init__(
        self,
        encoder,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self.encoder = encoder
        # self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        # self.net1 = cnn_extractor(dim = 3, input_plane = 16)
        # self.net2 = Transformer()
        # self.net = torch.optim.swa_utils.AveragedModel(self._net)
        # self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0

        # self.Seq_Transformer = Seq_Transformer(patch_size=127, dim=64, depth=4, heads=4, mlp_dim=64)
    
    # def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
    #     ''' Training the TS2Vec model.
        
    #     Args:
    #         train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
    #         n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
    #         n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
    #         verbose (bool): Whether to print the training loss after each epoch.
            
    #     Returns:
    #         loss_log: a list containing the training losses on each epoch.
    #     '''
    #     assert train_data.ndim == 3
        
    #     if n_iters is None and n_epochs is None:
    #         n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
    #     if self.max_train_length is not None:
    #         sections = train_data.shape[1] // self.max_train_length
    #         if sections >= 2:
    #             train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

    #     temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    #     if temporal_missing[0] or temporal_missing[-1]:
    #         train_data = centerize_vary_length_series(train_data)
                
    #     train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
    #     train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
    #     train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
    #     optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
    #     loss_log = []
        
    #     while True:
    #         if n_epochs is not None and self.n_epochs >= n_epochs:
    #             break
            
    #         cum_loss = 0
    #         n_epoch_iters = 0
            
    #         interrupted = False
    #         for batch in train_loader:
    #             if n_iters is not None and self.n_iters >= n_iters:
    #                 interrupted = True
    #                 break
                
    #             x = batch[0]
    #             if self.max_train_length is not None and x.size(1) > self.max_train_length:
    #                 window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
    #                 x = x[:, window_offset : window_offset + self.max_train_length]
    #             x = x.to(self.device)
                
    #             ts_l = x.size(1)
    #             crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
    #             crop_left = np.random.randint(ts_l - crop_l + 1)
    #             crop_right = crop_left + crop_l
    #             crop_eleft = np.random.randint(crop_left + 1)
    #             crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
    #             crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
    #             optimizer.zero_grad()
                
    #             out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
    #             out1 = out1[:, -crop_l:]
                
    #             out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
    #             out2 = out2[:, :crop_l]

    #             # print("out1", out1.shape, "out2", out2.shape)
                
    #             loss = hierarchical_contrastive_loss(
    #                 out1,
    #                 out2,
    #                 temporal_unit=self.temporal_unit
    #             )
                
    #             loss.backward()
    #             optimizer.step()
    #             # self.net.update_parameters(self._net)
                    
    #             cum_loss += loss.item()
    #             n_epoch_iters += 1
                
    #             self.n_iters += 1
                
    #             if self.after_iter_callback is not None:
    #                 self.after_iter_callback(self, loss.item())
            
    #         if interrupted:
    #             break
            
    #         cum_loss /= n_epoch_iters
    #         loss_log.append(cum_loss)
    #         if verbose:
    #             print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
    #         self.n_epochs += 1
            
    #         if self.after_epoch_callback is not None:
    #             self.after_epoch_callback(self, cum_loss)
            
    #     return loss_log
    
    def forward(self, x): # add this to form the modular code 
        # optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        x = x.permute(0,2,1) #(B,T,dim)

        if self.max_train_length is not None and x.size(1) > self.max_train_length:
            window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
            x = x[:, window_offset : window_offset + self.max_train_length]
        
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        

        in_1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        in_1 = in_1.permute(0,2,1)  #(B, dim, T)
        out1 = self.encoder(in_1)
        out1 = out1.permute(0,2,1) #(B, T, dim)

        in_2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        in_2 = in_2.permute(0,2,1)  #(B, dim, T)
        out2 = self.encoder(in_2)
        out2 = out2.permute(0,2,1) #(B, T, dim)

        # out1 = out1[:, -crop_l:, :]
        # out2 = out2[:, :crop_l, :]
        crop_l_new = min(out1.shape[1], out2.shape[1])
        out1 = out1[:, -crop_l_new:, :]
        out2 = out2[:, :crop_l_new, :]


        
        
        loss = hierarchical_contrastive_loss(
            out1,
            out2,
            temporal_unit=self.temporal_unit
        )
        
        del out1, out2
        torch.cuda.empty_cache()
        return loss
                
    # def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
    #     out = self.net(x.to(self.device, non_blocking=True), mask)
        
    #     if encoding_window == 'full_series':
    #         if slicing is not None:
    #             out = out[:, slicing]
    #         out = F.max_pool1d(
    #             out.transpose(1, 2),
    #             kernel_size = out.size(1),
    #         ).transpose(1, 2)
            
    #     elif isinstance(encoding_window, int):
    #         out = F.max_pool1d(
    #             out.transpose(1, 2),
    #             kernel_size = encoding_window,
    #             stride = 1,
    #             padding = encoding_window // 2
    #         ).transpose(1, 2)
    #         if encoding_window % 2 == 0:
    #             out = out[:, :-1]
    #         if slicing is not None:
    #             out = out[:, slicing]
            
    #     elif encoding_window == 'multiscale':
    #         p = 0
    #         reprs = []
    #         while (1 << p) + 1 < out.size(1):
    #             t_out = F.max_pool1d(
    #                 out.transpose(1, 2),
    #                 kernel_size = (1 << (p + 1)) + 1,
    #                 stride = 1,
    #                 padding = 1 << p
    #             ).transpose(1, 2)
    #             if slicing is not None:
    #                 t_out = t_out[:, slicing]
    #             reprs.append(t_out)
    #             p += 1
    #         out = torch.cat(reprs, dim=-1)
            
    #     else:
    #         if slicing is not None:
    #             out = out[:, slicing]
    #     return out
    #     # return out.cpu()
    
    # def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
    #     x = data

    #     out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
    #     if encoding_window == 'full_series':
    #         out = out.squeeze(1)
                        
    #     return out
         
    # def save(self, fn):
    #     ''' Save the model to a file.
        
    #     Args:
    #         fn (str): filename.
    #     '''
    #     torch.save(self.net.state_dict(), fn)
    
    # def load(self, fn):
    #     ''' Load the model from a file.
        
    #     Args:
    #         fn (str): filename.
    #     '''
    #     state_dict = torch.load(fn, map_location=self.device)
    #     self.net.load_state_dict(state_dict)
    

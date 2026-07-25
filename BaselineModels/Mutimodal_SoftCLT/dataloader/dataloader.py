import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from tslearn.metrics import dtw, dtw_path,gak
# import numpy as np
from tslearn.metrics import cdist_dtw
from pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW

import torch
from torch.utils.data import Dataset

# from .augmentations import DataTransform
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import torch.nn.functional as F

def get_DTW(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = dtw(UTS_tr[i].reshape(-1,1), UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_TAM(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                k = dtw_path(UTS_tr[i].reshape(-1,1), 
                             UTS_tr[j].reshape(-1,1))[0]
                p = [np.array([i[0] for i in k]),
                     np.array([i[1] for i in k])]
                dist = tam(p)
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_GAK(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = gak(UTS_tr[i].reshape(-1,1), 
                           UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat


# def get_MDTW(MTS_tr):
#     N = MTS_tr.shape[0]
#     dist_mat = np.zeros((N,N))
#     for i in tqdm.tqdm(range(N), miniters=100):
#         for j in range(N):
#             if i>j:
#                 mdtw_dist = dtw(MTS_tr[i], MTS_tr[j])
#                 dist_mat[i,j] = mdtw_dist
#                 dist_mat[j,i] = mdtw_dist
#             elif i==j:
#                 dist_mat[i,j] = 0
#             else:
#                 pass
    
#     return dist_mat



# def get_MDTW(MTS_tr, sakoe_chiba_radius=None, n_jobs=1):
#     """
#     MTS_tr: array-like, shape (N, T, d) or (N, T)
#     """
#     dist_mat = cdist_dtw(
#         MTS_tr,
#         global_constraint="sakoe_chiba" if sakoe_chiba_radius is not None else None,
#         sakoe_chiba_radius=sakoe_chiba_radius,
#         n_jobs=n_jobs
#     )
#     # dist_mat.fill_diagonal_(0)
#     return dist_mat

# def get_MDTW(MTS_tr, sakoe_chiba_radius=None, n_jobs=1, block_size=100):
#     N = MTS_tr.shape[0]
#     D = np.zeros((N, N), dtype=np.float32)

#     for i in tqdm.trange(0, N, block_size, desc="DTW blocks"):
#         j_end = min(i + block_size, N)

#         D_block = cdist_dtw(
#             MTS_tr[i:j_end],
#             MTS_tr[i:],
#             global_constraint="sakoe_chiba" if sakoe_chiba_radius is not None else None,
#             sakoe_chiba_radius=sakoe_chiba_radius,
#             n_jobs=n_jobs
#         )

#         for bi, ii in enumerate(range(i, j_end)):
#             D[ii, ii:] = D_block[bi]
#             D[ii:, ii] = D_block[bi]

#     return D

# def softdtw_pairwise_blocked(x, sdtw, block=16):
#     B, T, C = x.shape
#     D = torch.zeros(B, B, device=x.device)

#     for i in tqdm(range(0, B, block)):
#         bi = min(block, B - i)
#         xi = x[i:i+bi]

#         for j in range(0, B, block):
#             bj = min(block, B - j)
#             xj = x[j:j+bj]

#             xi_ = xi.unsqueeze(1).expand(bi, bj, T, C)
#             xj_ = xj.unsqueeze(0).expand(bi, bj, T, C)

#             xi_ = xi_.reshape(-1, T, C)
#             xj_ = xj_.reshape(-1, T, C)

#             D[i:i+bi, j:j+bj] = sdtw(xi_, xj_).view(bi, bj)

#             # crucial for memory
#             del xi_, xj_
#             torch.cuda.empty_cache()

#     return D
@torch.no_grad()
def softdtw_pairwise_blocked(x, sdtw, block=16): 
    """
    x: (B, T, C) tensor on GPU
    """
    B, T, C = x.shape
    
    # FIX 1: Allocate the huge matrix on CPU (System RAM), not GPU
    # We use float32 to save space (float64 would double the size)
    D = torch.zeros(B, B, dtype=torch.float16, device='cpu')

    # Create a progress bar
    print(f"Computing {B}x{B} matrix in blocks of {block}...")
    with torch.no_grad():
        for i in tqdm(range(0, B, block)):
            bi = min(block, B - i)
            # Slice on GPU
            xi = x[i : i + bi] 

            for j in range(0, B, block):
                bj = min(block, B - j)
                xj = x[j : j + bj]

                # Prepare inputs for SoftDTW (broadcasting)
                # These tensors live on GPU
                xi_ = xi.unsqueeze(1).expand(bi, bj, T, C)
                xj_ = xj.unsqueeze(0).expand(bi, bj, T, C)
                
                # Flatten for the batch processor of SoftDTW
                xi_ = xi_.reshape(-1, T, C)
                xj_ = xj_.reshape(-1, T, C)

                # Compute on GPU
                # Result is (bi * bj)
                distances = sdtw(xi_, xj_)
                
                # Reshape to (bi, bj)
                distances = distances.view(bi, bj)

                # FIX 2: Move the result chunk to CPU immediately and assign
                D[i : i + bi, j : j + bj] = distances.cpu()

                # Clean up GPU memory for the next iteration
                del xi_, xj_, distances
                
            # Optional: Empty cache periodically (e.g., every row block)
            torch.cuda.empty_cache()

    return D.numpy()

def shrink_sequence(x, target_length=None, pooling_factor=None):
    """
    Shrinks the time dimension of x.
    Args:
        x: Input tensor (Batch, Time, Channels) or (Batch, Channels, Time)
        target_length: Force sequence to this specific length (e.g., 100)
        pooling_factor: Reduce length by this factor (e.g., 4)
    """
    # 1. Detect format (Batch, Time, Channels) vs (Batch, Channels, Time)
    # We assume the smallest dimension is Channels, largest is Time
    if x.shape[1] > x.shape[2]: 
        # (Batch, Time, Channels) -> Permute to (Batch, Channels, Time) for pooling
        x = x.permute(0, 2, 1)
        permuted = True
    else:
        permuted = False
        
    B, C, T = x.shape
    
    # 2. Calculate Kernel Size (Stride)
    if target_length is not None:
        # adaptive_avg_pool1d forces exact output size
        x_shrunk = F.adaptive_avg_pool1d(x, target_length)
    elif pooling_factor is not None:
        # avg_pool1d reduces by factor
        x_shrunk = F.avg_pool1d(x, kernel_size=pooling_factor, stride=pooling_factor)
    else:
        return x # No shrinking

    # 3. Restore original shape
    if permuted:
        x_shrunk = x_shrunk.permute(0, 2, 1)
        
    print(f" shrunk sequence from {T} -> {x_shrunk.shape[1] if permuted else x_shrunk.shape[2]}")
    return x_shrunk

def batch_cosine_dist_gpu(x, block_size=1024):
    """
    Computes Pairwise Cosine DISTANCE (1 - CosineSim) efficiently.
    x: (B, T, C) or (B, D, T)
    Returns: (B, B) matrix on CPU
    """
    # 1. Standardize Input Shape to (Batch, Features)
    # We flatten Time and Channels into a single vector per sample
    if x.ndim == 3:
        B = x.shape[0]
        x = x.reshape(B, -1) # Flatten: (B, T*C)
    else:
        B = x.shape[0]

    print(f"Input shape flattened to: {x.shape}")

    # 2. Normalize vectors (L2 Norm)
    # This allows us to use simple Matrix Multiplication later
    # CosineSim(A, B) = (A . B) / (|A|*|B|)
    # If |A|=1 and |B|=1, then CosineSim(A, B) = A . B
    print("Normalizing vectors...")
    x = F.normalize(x, p=2, dim=1)
    
    # Move input to GPU (Usually fits: 120k * 3000 * 4 bytes ~= 1.4 GB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    # 3. Allocate Result Matrix on CPU (System RAM) to avoid GPU OOM
    # 120,000 * 120,000 * 4 bytes = ~53 GB
    # If you have <64GB RAM, consider using np.memmap (Disk) here instead.
    dist_mat = torch.zeros(B, B, dtype=torch.float16, device='cpu')

    print(f"Computing Cosine Distance Matrix ({B}x{B})...")
    
    # 4. Blocked Matrix Multiplication
    # We compute 'x @ x.T' in blocks
    with torch.no_grad():
        for i in tqdm(range(0, B, block_size)):
            end_i = min(i + block_size, B)
            # Slice row block
            x_i = x[i:end_i] 

            for j in range(0, B, block_size):
                end_j = min(j + block_size, B)
                # Slice column block
                x_j = x[j:end_j]
                
                # Compute Dot Product (Similarity)
                # Shape: (block_i, block_j)
                sim_block = torch.mm(x_i, x_j.T)
                
                # Convert Similarity to Distance (1 - Sim)
                # Clamp is used to avoid tiny floating point errors (e.g., 1.0000001)
                dist_block = 1 - sim_block.clamp(min=-1.0, max=1.0)
                
                # Move to CPU and assign
                dist_mat[i:end_i, j:end_j] = dist_block.cpu()
                
                del sim_block, dist_block
            
            # Optional: Clear cache if GPU memory is tight
            # torch.cuda.empty_cache()

    return dist_mat.numpy()

def batch_soft_dtw_gpu(x , gamma=0.1):
    """
    x: (B, D, T) torch tensor on CUDA
    returns: (B, B) distance matrix
    """
    # N, D, T = x.shape
    # print(f"TYPE: {type(x)}")
    # print(f"SHAPE: {x.shape}")
    # print(f"NDIM: {getattr(x, 'ndim', 'N/A')}")
    # x = x.transpose(1, 2)  # (B, T, D)
    # x = x.transpose(0, 2, 1)
    x = torch.tensor(x, dtype=torch.float16)
    # x = shrink_sequence(x, target_length=300)  # Shrink to 1000 time steps if needed
    x = x.to('cuda')
    print(x.shape[1])
    if x.shape[1] > 1000: # sleepedfx
        # print("Using bandwidth=10 for SoftDTW due to long sequence length")
        # sdtw = SoftDTW(use_cuda=True, gamma=gamma, bandwidth=10)
        mdtw_dist = batch_cosine_dist_gpu(x)
        print(mdtw_dist)
    else:
        sdtw = SoftDTW(use_cuda=True, gamma=gamma)
        mdtw_dist = softdtw_pairwise_blocked(x, sdtw)
    np.fill_diagonal(mdtw_dist, 0)
    
    # dist_mat = mdtw_dist.fill_diagonal_(0)
    # dist_mat = dist_mat.to(x.device)
    # return dist_mat.numpy()
    return mdtw_dist




def get_COS(MTS_tr):
    cos_sim_matrix = -cosine_similarity(MTS_tr)
    return cos_sim_matrix

def get_EUC(MTS_tr):
    return euclidean_distances(MTS_tr)

def save_sim_mat(X_tr, min_ = 0, max_ = 1, multivariate=False, type_='DTW'):
    N = X_tr.shape[0]
    # print('X_tr shape:', X_tr.shape)
    if multivariate:
        assert type_=='DTW'
        # dist_mat = get_MDTW(X_tr, sakoe_chiba_radius=10, n_jobs=4)
        dist_mat = batch_soft_dtw_gpu(X_tr)
    else:
        if type_=='DTW':
            dist_mat = get_DTW(X_tr)
        elif type_=='TAM':
            dist_mat = get_TAM(X_tr)
        elif type_=='COS':
            dist_mat = get_COS(X_tr)
        elif type_=='EUC':
            dist_mat = get_EUC(X_tr)
        elif type_=='GAK':
            dist_mat = get_GAK(X_tr)
        
    # # (1) distance matrix
    # diag_indices = np.diag_indices(N)
    # mask = np.ones(dist_mat.shape, dtype=bool)
    # mask[diag_indices] = False
    # temp = dist_mat[mask].reshape(N, N-1)
    # dist_mat[diag_indices] = temp.min()
    
    # # (2) normalized distance matrix
    # scaler = MinMaxScaler(feature_range=(min_, max_))
    # dist_mat = scaler.fit_transform(dist_mat)
    
    # # (3) normalized similarity matrix
    # return 1 - dist_mat 
    
    print("Post-processing matrix (Memory Efficient Mode)...")
    N = dist_mat.shape[0]

    # --- STEP 1: Handle Diagonal (Faithful Re-implementation) ---
    # Original logic: Set diagonal to the minimum value of the non-diagonal elements.
    # Optimization: We do this without creating a huge mask copy.
    
    # A. Temporarily fill diagonal with Infinity so it doesn't affect the min calculation
    np.fill_diagonal(dist_mat, np.inf)
    
    # B. Find the global min of the matrix (ignoring the Inf diagonal)
    # This matches 'temp.min()' from your original code
    min_val = dist_mat.min()
    
    # C. Set the diagonal to this minimum value
    np.fill_diagonal(dist_mat, min_val)
    
    print(f"Diagonal filled with global min: {min_val}")


    # --- STEP 2: Normalization (Faithful MinMaxScaler Re-implementation) ---
    # Original logic: scaler.fit_transform(dist_mat) -> Normalizes each COLUMN (feature).
    # Optimization: We loop through columns in blocks to perform the math in-place.
    
    print("Applying MinMaxScaler in blocks (Column-wise)...")
    
    # Block size for processing columns (adjust based on RAM, 1024 is safe)
    col_block_size = 1024
    
    # Pre-calculate the scale range
    feature_range = max_ - min_
    
    for i in tqdm(range(0, N, col_block_size)):
        # Define the slice for this block of columns
        end_i = min(i + col_block_size, N)
        
        # 1. Slice the columns (creates a small temporary copy in RAM)
        col_chunk = dist_mat[:, i:end_i]
        
        # 2. Compute Min and Max for these columns
        # axis=0 means min/max along the column
        c_min = col_chunk.min(axis=0)
        c_max = col_chunk.max(axis=0)
        
        # Avoid division by zero (if max == min, denominator becomes 1)
        denom = c_max - c_min
        denom[denom == 0] = 1.0
        
        # 3. Apply MinMax formula: (X - min) / (max - min)
        # We do this in-place on the chunk
        col_chunk -= c_min
        col_chunk /= denom
        
        # 4. Scale to target range (min_, max_)
        if feature_range != 1.0:
            col_chunk *= feature_range
        if min_ != 0:
            col_chunk += min_
            
        # 5. Write back to the main matrix
        dist_mat[:, i:end_i] = col_chunk
        
        # Flush to disk periodically if using memmap
        if isinstance(dist_mat, np.memmap) and i % (col_block_size * 5) == 0:
            dist_mat.flush()

    # --- STEP 3: Inversion (Faithful Re-implementation) ---
    # Original logic: return 1 - dist_mat
    # Optimization: In-place arithmetic
    
    print("Inverting to Similarity...")
    # dist_mat = -dist_mat + 1
    # We do this in-place to avoid creating a new 53GB array
    dist_mat *= -1
    dist_mat += 1
    
    if isinstance(dist_mat, np.memmap):
        dist_mat.flush()

    return dist_mat

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return index, self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return index, self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode, pc, batch_size):
    #batch_size = configs.batch_size

    if training_mode != "SupCon":
        if ('ft' in training_mode) & (pc == 1):
            print('1%')
            train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"))
        elif ('ft' in training_mode) & (pc == 5):
            print('5%')
            train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
        elif ('ft' in training_mode) & (pc == 10):
            print('10%')
            train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
        elif ('ft' in training_mode) & (pc == 50):
            print('50%')
            train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
        elif ('ft' in training_mode) & (pc == 75):
            print('75%')
            train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
        else:
            train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    else :
        train_dataset = torch.load(os.path.join(data_path, f"pseudo_train_data_{str(pc)}perc.pt"))
    
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    print('Train size: ',train_dataset['samples'].shape)
    print('Valid size: ',valid_dataset['samples'].shape)
    print('Test size: ',test_dataset['samples'].shape)
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    if batch_size == 999:
        if train_dataset.__len__() < batch_size:
            if train_dataset.__len__() > 16:
                batch_size = 16
            else:
                batch_size = 4
        else:
            batch_size = configs.batch_size

    print('batch_size',batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    return train_loader, valid_loader, test_loader


def data_generator_wo_val(data_path, configs, training_mode, pc, batch_size):
    #batch_size = configs.batch_size

    if training_mode != "SupCon":
        if ('ft' in training_mode) & (pc == 1):
            print('1%')
            train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"))
        elif ('ft' in training_mode) & (pc == 5):
            print('5%')
            train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
        elif ('ft' in training_mode) & (pc == 10):
            print('10%')
            train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
        elif ('ft' in training_mode) & (pc == 50):
            print('50%')
            train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
        elif ('ft' in training_mode) & (pc == 75):
            print('75%')
            train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
        else:
            train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    else :
        train_dataset = torch.load(os.path.join(data_path, f"pseudo_train_data_{str(pc)}perc.pt"))
    
#    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    train_dataset['samples'] = train_dataset['samples']#[:1000,:,:]
    test_dataset['samples'] = test_dataset['samples']#[:1000,:,:]
    train_dataset['labels'] = train_dataset['labels']#[:1000]
    test_dataset['labels'] = test_dataset['labels']#[:1000]
    print('Train size: ',train_dataset['samples'].shape)
    print('Test size: ',test_dataset['samples'].shape)
    
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    if train_dataset.__len__() < configs.batch_size:
        if train_dataset.__len__() > 16:
            batch_size = 16
        else:
            batch_size = 4
    else:
        batch_size = configs.batch_size
            
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    return train_loader,test_loader

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def normalize_TS(TS):
    TS = set_nan_to_zero(TS)
    if TS.ndim == 2:
        print('Preprocessing UTS ...')
        TS_max = TS.max(axis = 1).reshape(-1,1)
        TS_min = TS.min(axis = 1).reshape(-1,1)
        TS = (TS - TS_min)/(TS_max - TS_min + (1e-6))        
    elif TS.ndim == 3:
        print('Preprocessing MTS ...')
        N, D, L = TS.shape
        TS_max = TS.max(axis=2).reshape(N,D,1) 
        TS_min = TS.min(axis=2).reshape(N,D,1)
        TS = (TS - TS_min) / (TS_max - TS_min + (1e-6))   
    return TS


def save_cosine_sim_float16(x, save_path, min_ = 0, max_ = 1, block_size=512):
    """
    Computes Cosine Similarity matrix (N x N) in chunks.
    Saves to disk as FLOAT16 (2 bytes) to save 50% space (26GB vs 53GB).
    Applies Diagonal Fix -> MinMax Norm -> Inversion on the fly.
    """
    # 1. Prepare Input (Flatten & Normalize)
    if x.ndim == 3: x = x.reshape(x.shape[0], -1)
    B = x.shape[0]
    N = B
    
    print(f"Normalizing input {x.shape}...")
    x = torch.tensor(x, dtype=torch.float32)
    # Normalize inputs so Dot Product == Cosine Similarity
    x = F.normalize(x, p=2, dim=1).cuda()
    # x = F.normalize(x, p=2, dim=1)

    # 2. Create the Memory-Mapped File (26.8 GB)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Creating float16 matrix at {save_path}...")
    
    # Keep the .npy header so later np.load(..., mmap_mode=...) can reopen it.
    dist_mat = np.lib.format.open_memmap(save_path, dtype='float16', mode='w+', shape=(N, N))

    print("Computing Cosine Similarity & Post-processing in blocks...")
    
    with torch.no_grad():
        for i in tqdm(range(0, B, block_size)):
            end_i = min(i + block_size, B)
            # Slice row block
            x_i = x[i:end_i] 

            for j in range(0, B, block_size):
                end_j = min(j + block_size, B)
                # Slice column block
                x_j = x[j:end_j]
                
                # Compute Dot Product (Similarity)
                # Shape: (block_i, block_j)
                sim_block = torch.mm(x_i, x_j.T)
                
                # Convert Similarity to Distance (1 - Sim)
                # Clamp is used to avoid tiny floating point errors (e.g., 1.0000001)
                dist_block = 1 - sim_block.clamp(min=-1.0, max=1.0)
                
                # Move to CPU and assign
                dist_mat[i:end_i, j:end_j] = dist_block.cpu()
                
                del sim_block, dist_block
    
    print("Post-processing matrix (Memory Efficient Mode)...")
    N = dist_mat.shape[0]

    # --- STEP 1: Handle Diagonal (Faithful Re-implementation) ---
    # Original logic: Set diagonal to the minimum value of the non-diagonal elements.
    # Optimization: We do this without creating a huge mask copy.
    
    # A. Temporarily fill diagonal with Infinity so it doesn't affect the min calculation
    np.fill_diagonal(dist_mat, np.inf)
    
    # B. Find the global min of the matrix (ignoring the Inf diagonal)
    # This matches 'temp.min()' from your original code
    min_val = dist_mat.min()
    
    # C. Set the diagonal to this minimum value
    np.fill_diagonal(dist_mat, min_val)
    
    print(f"Diagonal filled with global min: {min_val}")


    # --- STEP 2: Normalization (Faithful MinMaxScaler Re-implementation) ---
    # Original logic: scaler.fit_transform(dist_mat) -> Normalizes each COLUMN (feature).
    # Optimization: We loop through columns in blocks to perform the math in-place.
    
    print("Applying MinMaxScaler in blocks (Column-wise)...")
    
    # Block size for processing columns (adjust based on RAM, 1024 is safe)
    col_block_size = 1024
    
    # Pre-calculate the scale range
    feature_range = max_ - min_
    
    for i in tqdm(range(0, N, col_block_size)):
        # Define the slice for this block of columns
        end_i = min(i + col_block_size, N)
        
        # 1. Slice the columns (creates a small temporary copy in RAM)
        col_chunk = dist_mat[:, i:end_i]
        
        # 2. Compute Min and Max for these columns
        # axis=0 means min/max along the column
        c_min = col_chunk.min(axis=0)
        c_max = col_chunk.max(axis=0)
        
        # Avoid division by zero (if max == min, denominator becomes 1)
        denom = c_max - c_min
        denom[denom == 0] = 1.0
        
        # 3. Apply MinMax formula: (X - min) / (max - min)
        # We do this in-place on the chunk
        col_chunk -= c_min
        col_chunk /= denom
        
        # 4. Scale to target range (min_, max_)
        if feature_range != 1.0:
            col_chunk *= feature_range
        if min_ != 0:
            col_chunk += min_
            
        # 5. Write back to the main matrix
        dist_mat[:, i:end_i] = col_chunk
        
        # Flush to disk periodically if using memmap
        if isinstance(dist_mat, np.memmap) and i % (col_block_size * 5) == 0:
            dist_mat.flush()

    # --- STEP 3: Inversion (Faithful Re-implementation) ---
    # Original logic: return 1 - dist_mat
    # Optimization: In-place arithmetic
    
    print("Inverting to Similarity...")
    # dist_mat = -dist_mat + 1
    # We do this in-place to avoid creating a new 53GB array
    dist_mat *= -1
    dist_mat += 1
    


    dist_mat.flush()
    print(f"Saved float16 matrix to {save_path}")
    return dist_mat

def load_existing_sim_matrix(MAT_PATH, n_samples, dtype='float16'):
    try:
        return np.load(MAT_PATH, mmap_mode='r')
    except ValueError as exc:
        expected_bytes = n_samples * n_samples * np.dtype(dtype).itemsize
        actual_bytes = os.path.getsize(MAT_PATH)
        if actual_bytes != expected_bytes:
            raise
        print(f"Loading legacy raw memmap matrix at {MAT_PATH}")
        return np.memmap(MAT_PATH, dtype=dtype, mode='r', shape=(n_samples, n_samples))

def load_sim_matrix(dataset, train, MAT_PATH, type_='DTW'):
    # tr = torch.load(f'data/{dataset}/train.pt')
    # train = tr['samples'].detach().cpu().numpy().astype(np.float64)

    
    if (train.ndim==3) & (train.shape[1]==1):
        train=train.squeeze(1)
    elif (train.ndim==3) & (train.shape[1]>1):
        train=train.transpose(0,2,1)
    
    # os.makedirs(f'data/{dataset}', exist_ok=True)
    # MAT_PATH = os.path.join(f'data/{dataset}',f'{type_}.npy')
        
    if os.path.exists(MAT_PATH):
        print(f"{type_} already exists")
        sim_mat = load_existing_sim_matrix(MAT_PATH, train.shape[0])
    else:
        if type_ == 'DTW':
            print(f"Saving {type_} ...")
            sim_mat = save_sim_mat(normalize_TS(train), min_ = 0, max_ = 1, multivariate=True, type_=type_)
            print("finish computing, saving to disk ...")
            np.save(MAT_PATH, sim_mat)
            print("Finish saving.")
        else:
            print(f"Saving {type_} ...")
            sim_mat = save_cosine_sim_float16(normalize_TS(train), MAT_PATH, block_size=1024)
            print("Finish saving.")
        
    return sim_mat

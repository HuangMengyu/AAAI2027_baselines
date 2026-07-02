import numpy as np
import argparse
import random
import gc
from datetime import datetime

from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

from model import Model as tfcc_model
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
while _PROJECT_ROOT and not os.path.isdir(os.path.join(_PROJECT_ROOT, 'common_utils')):
    _PARENT = os.path.dirname(_PROJECT_ROOT)
    if _PARENT == _PROJECT_ROOT:
        break
    _PROJECT_ROOT = _PARENT
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common_utils.dataset_loader import get_path_loader_new, get_idx, get_loader
from base_models import SSLDataSet, FTDataSet

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import os

# def simple_collate(batch):
#     # batch: list of tensors [tensor([3000,3]), ...]
#     return torch.stack(batch, dim=0)

def self_supervised_learning(fold, model, X, n_epoch, lr, batch_size, device, model_save_path, min_ratio=0.3, max_ratio=0.8, train_with_iter=False, num_iter=600):
    # optimizer = optim.Adam(model.parameters(), lr)
    start_datetime = datetime.now()
    optimizer = optim.AdamW(model.parameters(), lr, betas=(0.5, 0.99), weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.to(device)
    model.train()

    dataset = SSLDataSet(X)
    print('SSL dataset:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    

    losses = []
    # torch.autograd.set_detect_anomaly(True)
    if not train_with_iter:
        pbar = trange(n_epoch)
        for _ in pbar:
            # num = 0
            # for batch in tqdm(dataloader):
            for batch in dataloader:
                time, freq, time_aug, freq_aug = batch
                time = time.to(device)
                freq = freq.to(device)
                time_aug = time_aug.to(device)
                freq_aug = freq_aug.to(device)
                x = (time, freq, time_aug, freq_aug)
                loss = model(x, ssl=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
                # num += 1
                # if num == 1:
                #     break
            scheduler.step(_)
            pbar.set_description(str(sum(losses) / len(losses)))
    else:
        print("Training with iterations...")
        pbar = trange(num_iter)
        dataloader_iterator = iter(dataloader)
        for _ in pbar:
            # num = 0
            # for batch in tqdm(dataloader):
            batch = next(dataloader_iterator)
            time, freq, time_aug, freq_aug = batch
            time = time.to(device)
            freq = freq.to(device)
            time_aug = time_aug.to(device)
            freq_aug = freq_aug.to(device)
            x = (time, freq, time_aug, freq_aug)
            loss = model(x, ssl=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # losses.append(float(loss))
            # num += 1
            # if num == 1:
            #     break
            scheduler.step(_)
            # pbar.set_description(str(sum(losses) / len(losses)))
            pbar.set_description(str(float(loss)))

    torch.save(model.to('cpu'), os.path.join(model_save_path, f'Pretrained_Model_{fold}.pkl'))
    end_datetime = datetime.now()
    time_diff = end_datetime - start_datetime
    time_diff_seconds = time_diff.total_seconds()
    print(f"Pretraining Time for fold {fold}: {time_diff_seconds} seconds")

def finetuning(fold, model, train_set, valid_set, n_epoch, lr, device, model_save_path, batch_size=128, multi_label=True, frozen_backbone=False, n_classes=5):
    # multi_label: whether the classification task is a multi-label task.
    start_datetime = datetime.now()
    model.train()
    model.to(device)
    
    print("Fine-tuning train set:", len(train_set))
    print("Fine-tuning valid set:", len(valid_set))
    print("Batch size:", batch_size)
    print("n_epoch:", n_epoch)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    
    # for stage in range(2):
        # stage0: finetuning only classifier; stage1: finetuning whole model
    best_auc = 0
    step = 0
    # if stage == 0:
    #     min_lr = 1e-6
    #     optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    # else:
    min_lr = 1e-8
    # optimizer = optim.Adam(model.parameters(), lr=lr/2)
    if not frozen_backbone:
        optimizer = optim.AdamW(model.parameters(), lr, betas=(0.5, 0.99), weight_decay=3e-4)
    elif frozen_backbone:
        optimizer = optim.AdamW(model.classifier.parameters(), lr, betas=(0.5, 0.99), weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode = 'max', factor=0.8, min_lr=min_lr)
    pbar = trange(n_epoch)
    for _ in pbar:
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            x, y = tuple(t.to(device) for t in batch)
            pred = model(x)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if step == 1:
            #     break
            # if step % 60 == 0:
            gap = max(1, len(train_loader) // 5)
            if step % gap == 0 :
                valid_auc, val_label = val(model, valid_set, batch_size, multi_label, n_classes)
                pbar.set_description('Best Validation AUC: {:.4f} --------- AUC on this step: {:.4f}'.format(best_auc, valid_auc))
                if valid_auc > best_auc:
                    best_auc = valid_auc
                    if not frozen_backbone:
                        torch.save(model, os.path.join(model_save_path, f'Finetuned_Model_{fold}.pkl'))
                    elif frozen_backbone:
                        torch.save(model, os.path.join(model_save_path, f'Finetuned_Model_{fold}_frozenbackbone.pkl'))
                scheduler.step(best_auc)
    end_datetime = datetime.now()
    time_diff = end_datetime - start_datetime
    time_diff_seconds = time_diff.total_seconds()
    print(f"Pretraining Time for fold {fold}: {time_diff_seconds} seconds")

def multiclass_auc_with_all_classes(y_true, y_score, n_classes=8):
    y_score = np.array(y_score)
    
    aucs = []
    for c in range(n_classes):
        if np.any(y_true == c):  # class present
            auc = roc_auc_score((y_true == c).astype(int), y_score[:, c])
        else:  # class missing
            auc = np.nan
        aucs.append(auc)
    return aucs, np.nanmean(aucs)  # mean across all classes

def test(model, dataset, batch_size, multi_label, n_classes):
    print("Start testing...")
    model.to(device)
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)

    pred_prob = []
    # label = []
    with torch.no_grad():
        # num = 0
        # for batch in tqdm(testloader):
        for batch in testloader:
            x, y = tuple(t.to(device) for t in batch)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])
            # label.extend([y.cpu().detach().numpy().tolist()])
            # num += 1
            # if num == 1:
            #     break

    # label = np.array(label)
    # label = np.reshape(label, (-1,))
    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes=n_classes)
    # _, auc = multiclass_auc_with_all_classes(label, pred_prob, n_classes=5)
    
    pred_label = np.argmax(pred_prob, axis=1)
    F1 = f1_score(dataset.label, pred_label, average='macro', )
    acc = balanced_accuracy_score(dataset.label, pred_label, )
    # F1 = f1_score(label, pred_label, average='macro', )
    # acc = accuracy_score(label, pred_label, )
    
    print('AUC is {:.2f}'.format(auc * 100))
    print('F1 is', F1)
    print('acc is', acc)
    print('More metrics can be added in the test function.')
    model.train()
    return auc, F1, acc, dataset.label

def val(model, dataset, batch_size, multi_label, n_classes):
    model.to(device)
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)

    pred_prob = []
    label = []
    with torch.no_grad():
        # num = 0
        # for batch in tqdm(testloader):
        for batch in testloader:
            x, y = tuple(t.to(device) for t in batch)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])
            # label.extend([y.cpu().detach().numpy().tolist()])
            # num += 1
            # if num == 1:
            #     break

    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes=n_classes)
    # _, auc = multiclass_auc_with_all_classes(label, pred_prob, n_classes=5)
    
    model.train()
    return auc, dataset.label

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", type=str2bool, default=False)
    parser.add_argument("--sl", type=str2bool, default=True)
    parser.add_argument("--load", type=str2bool, default=True)
    parser.add_argument("--test", type=str2bool, default=True)
    # all default values of parameters are for PTB-XL
    parser.add_argument("--seq_len", type=int, default=10000)
    parser.add_argument("--patch_len", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--in_dim", type=int, default=12)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, default='SleepEDFx', choices=['SleepEDFx', 'PAMAP2', 'UCI-HAR', 'UCI-HAR_total'])
   
    # parser.add_argument("--dataset_path", type=str, default=f'C:\\Users\\mengyuh\\OneDrive - Chalmers\\Research\\ComparativeStudy\\Codes\\datasets_CRT_10fold\\SleepEDFx\\SleepCassette')
    parser.add_argument("--model_save_path", type=str, default='.')
    # parser.add_argument("--num_fold", type=int, default=10)
    parser.add_argument("--current_num_fold", type=int, default=1)
    parser.add_argument("--seeds", nargs='*', type=list, default=[0, 20,40,60,80])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epochs_ft", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--train_with_iter', type=str2bool, default=False, help='Train with iterations instead of epochs (defaults to False)')
    parser.add_argument('--num_iter', type=int, default=600, help='Number of iterations for training when train_with_iter is True (defaults to 600)')
    parser.add_argument('--frozen_backbone', type=str2bool, default=False, help='Whether to freeze the backbone during finetuning (defaults to False)')
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    seq_len = opt.seq_len
    patch_len = opt.patch_len
    dim = opt.dim
    in_dim = opt.in_dim
    n_classes = opt.n_classes
    # dataset_path = opt.dataset_path
    root_save_path = opt.model_save_path

    train_with_iter = opt.train_with_iter
    print("Train with iterations:", train_with_iter)
    # if train_with_iter:
    num_iter = opt.num_iter

    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path, exist_ok=True)

    # n_epoch = 1
    # n_epoch = 300
    n_epoch = opt.epochs
    # n_epoch_ft = opt.epochs_ft
    n_epoch_ft = 10
    lr = opt.lr
    batch_size = opt.batch_size
    

    print("n_epoch: ", n_epoch)
    print("n_epoch_ft: ", n_epoch_ft)
    print('lr', lr)
    print("batch_size", batch_size)
    
    ssl = opt.ssl
    if not ssl:
        print("Only Finetuning!")
    
    frozen_backbone = opt.frozen_backbone
    print("Frozen backbone:", frozen_backbone)
    if frozen_backbone:
        print("Freezing the backbone during finetuning!")

    seeds = opt.seeds[:3]
    # seeds = opt.seeds
    print(seeds)

    

       # set dataset path and current fold
    dataset_name = opt.dataset_name
    small_classifier = False
    transformer_depth = 4
    if dataset_name == 'SleepEDFx':
        dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette'
        n_classes = 5
        mod1_dims = 2
        mod2_dims = 1
        in_dim = mod1_dims + mod2_dims
        time_steps = 50
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
    elif dataset_name == 'PAMAP2':
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2'
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_512'
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_200'
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_200_noOverlap'
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_preprocessed_normalized'
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_2s_windowed'
        # dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_10s_windowed'
        # n_classes = 12
        dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_normalized_9classes'
        
        n_classes = 9
        mod1_dims = 9
        mod2_dims = 9
        # time_steps = 10
        in_dim = mod1_dims + mod2_dims
        time_steps = 5
        # small_classifier = True
        # transformer_depth = 2
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
    elif dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        if dataset_name == 'UCI-HAR_total':
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR_total'
        else:
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR'
        n_classes = 6
        mod1_dims = 3
        mod2_dims = 3
        in_dim = mod1_dims + mod2_dims
        time_steps = 3
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
    
    print("Dataset path:", dataset_path)
    
    file_parser = dict()
    file_parser["filepath"] = dataset_path
    file_parser["Fold"] = opt.current_num_fold
    print("Current Fold: ", file_parser["Fold"])

    for seed in seeds:
        print(f'=================Current Seed: {seed}=================')
        set_seed(seed)
        model_save_path = os.path.join(root_save_path, str(seed))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        i = opt.current_num_fold

        path, path_name = get_path_loader_new(file_parser["filepath"], dataset_name=dataset_name)

        
        pretrain_test_auc = []
        pretrain_test_F1 = []
        pretrain_test_acc = []
        finetune_test_auc = []
        finetune_test_F1 = []
        finetune_test_acc = []
        
        # Data initialization
        # pretrain_data, finetune_train_data, finetune_train_label, finetune_val_data, finetune_val_label, test_data, test_label = get_loader(file_parser, path, path_name, dataset_name)
        pretrain_data, pretrain_label, finetune_train_data, finetune_train_label, finetune_val_data, finetune_val_label, test_data, test_label = get_loader(file_parser, path, path_name, dataset_name)
        
        
        # for certain method, the dataset should have channel as the last dimension
        # pretrain_data = np.transpose(pretrain_data, (0, 2, 1))
        # finetune_train_data = np.transpose(finetune_train_data, (0, 2, 1))
        # finetune_val_data = np.transpose(finetune_val_data, (0, 2, 1))
        # test_data = np.transpose(test_data, (0, 2, 1))
        # print(pretrain_data.shape, finetune_train_data.shape, finetune_val_data.shape, test_data.shape)
        
        # model initialization
        model = tfcc_model(
            input_dims=in_dim,
            output_dims=127,
            device=device,
            num_class=n_classes,
            final_out_channels = 128, 
            kernel_size = 25, 
            stride = 3, 
            timesteps=time_steps, 
            hidden_dim=64,
            batch_size=batch_size
        )
        
        if ssl:
            # Pretraining
            self_supervised_learning(i, model, pretrain_data, n_epoch, lr, batch_size, device, model_save_path, train_with_iter=train_with_iter, num_iter=num_iter)
            
        # Load the pretrained model
        model = torch.load(os.path.join(model_save_path, f'Pretrained_Model_{i}.pkl'), map_location=device)
        
        TestSet = FTDataSet(test_data, test_label, multi_label=False)
        auc, F1, acc, _ = test(model, TestSet, batch_size=100, multi_label=False, n_classes=n_classes)
        pretrain_test_auc.append(auc)
        pretrain_test_F1.append(F1)
        pretrain_test_acc.append(acc)

        # Finetuning
        TrainSet = FTDataSet(finetune_train_data, finetune_train_label, multi_label=False) # multi_label = True when the dataset is PTBXL
        ValidSet = FTDataSet(finetune_val_data, finetune_val_label, multi_label=False)
        finetuning(i, model, TrainSet, ValidSet, n_epoch_ft, lr, device, model_save_path, batch_size=128, multi_label=False, frozen_backbone=frozen_backbone, n_classes=n_classes)

        # Testing

        TestSet = FTDataSet(test_data, test_label, multi_label=False)

        if not frozen_backbone:
            model = torch.load(os.path.join(model_save_path, f'Finetuned_Model_{i}.pkl'), map_location=device)
        elif frozen_backbone:
            model = torch.load(os.path.join(model_save_path, f'Finetuned_Model_{i}_frozenbackbone.pkl'), map_location=device)
        auc1, F11, acc1, _ = test(model, TestSet, batch_size=100, multi_label=False, n_classes=n_classes)
        finetune_test_auc.append(auc1)
        finetune_test_F1.append(F11)
        finetune_test_acc.append(acc1)

        # print out the 10 fold results
        
        print(f"=================Fold {i} Seed {seed} Results=================")
        print(f"Pretraining Test AUC: {pretrain_test_auc}, F1: {pretrain_test_F1}, Acc: {pretrain_test_acc}")
        print(f"Finetuning Test AUC: {finetune_test_auc}, F1: {finetune_test_F1}, Acc: {finetune_test_acc}")
        print(f"===================================================")
        
    

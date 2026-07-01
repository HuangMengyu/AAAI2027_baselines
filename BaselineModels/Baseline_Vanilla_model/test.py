import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from common_utils.dataset_loader import get_path_loader_new, get_idx
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, balanced_accuracy_score
from base_models import FTDataSet
import argparse

from model import Model as tfcc_model

import torch.nn as nn
import torch.nn.functional as F

def multiclass_auc_with_all_classes(y_true, y_score, n_classes=5):
    y_score = np.array(y_score)
    aucs = []
    for c in range(n_classes):
        if np.any(y_true == c):  # class present
            auc = roc_auc_score((y_true == c).astype(int), y_score[:, c])
        else:  # class missing
            auc = np.nan
        aucs.append(auc)
    return aucs, np.nanmean(aucs)  # mean across all classes


def get_test_loader_batch(path_name, test_idx):
    test_data = []
    test_label = []

    for t_idx in test_idx:
        for i in range(len(path_name[t_idx])):
            temp_data = np.load(path_name[t_idx][i][0], mmap_mode='r')
            temp_label = np.load(path_name[t_idx][i][1], mmap_mode='r')
            test_data.append(temp_data)
            test_label.append(temp_label)
    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    assert test_data.shape[0] == test_label.shape[0], "test data and test size do not match!"
    print("Number of test samples:", test_data.shape)

    # test_loader = DataLoader(dataset=test_builder, batch_size=64, shuffle=True, num_workers=4)

    return test_data, test_label

def test(model, dataset, batch_size, multi_label, device, n_classes=5):
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)

    pred_prob = []
    with torch.no_grad():
        for batch in testloader:
            *modalities, y = tuple(t.to(device) for t in batch)
            x = tuple(modalities)
            pred = model(x)

            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])

    pred_prob = np.array(pred_prob)
        
    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes=n_classes )
    
    pred_label = np.argmax(pred_prob, axis=1)

    F1 = f1_score(dataset.label, pred_label, average='macro', )
    acc = balanced_accuracy_score(dataset.label, pred_label, )
    
    print('AUC is {:.2f}'.format(auc * 100))
    print('F1 is', F1)
    print('acc is', acc)
    model.train()
    return auc, F1, acc

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def main():
    # initialize the args
    parser = dict()
    parser["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser["Fold"] = 1
    in_parser = argparse.ArgumentParser()
    in_parser.add_argument('--dataset_name', type=str, default='SleepEDFx', choices=['SleepEDFx', 'PAMAP2', 'UCI-HAR', 'UCI-HAR_total'], help='dataset name')
    in_parser.add_argument('--checkpoint_path', type=str, default='/mimer/NOBACKUP/groups/naiss2025-22-1224/Codes/TFCC_badpositive/ExperimentLog/5fold_bp_A_25warm', help='checkpoint path for the model checkpoints')
    
    in_parser.add_argument('--final_out_channels', type=int, default=128)
    in_parser.add_argument('--output_dims', type=int, default=127)

    opt = in_parser.parse_args()
    small_classifier = False
    dataset_name = opt.dataset_name
    if dataset_name == 'SleepEDFx':
        input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette"
        n_classes = 5
        mod1_dims = 2
        mod2_dims = 1
        time_steps = 50
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
   
    elif dataset_name == 'PAMAP2':
        # input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2"
        # input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_512'
        # input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_200'
        # input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_200_noOverlap'
        # input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_preprocessed_normalized'
        # input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_2s_windowed"
        # input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_10s_windowed"
        input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_normalized_9classes'
        
        # n_classes = 12
        n_classes = 9
        mod1_dims = 9
        mod2_dims = 9
        time_steps = 5
        # small_classifier = True
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
   
    elif dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        if dataset_name == 'UCI-HAR_total':
            input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR_total'
        else:
            input_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR'
        n_classes = 6
        mod1_dims = 3
        mod2_dims = 3
        time_steps = 3
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
    
    checkpoint_path = opt.checkpoint_path
    print("Evaluating Checkpoint path:", checkpoint_path)
    
    # seeds = [0, 20, 40, 60, 80]
    seeds = [0, 20, 40]
    if dataset_name == 'PAMAP2':
        # num_folds = 8
        num_folds = 4
    else:
        num_folds = 5
    acc_results = []
    f1_results = []
    auc_results = []
    device = parser["device"]

    model = tfcc_model(
            mod1_dims=mod1_dims,
            mod2_dims=mod2_dims,
            output_dims=opt.output_dims,
            device=device,
            num_class=n_classes,
            final_out_channels = opt.final_out_channels, 
            kernel_size = 25, 
            stride = 3, 
            timesteps=time_steps, 
            hidden_dim=64,
        ).to(device)

    for seed in seeds:
        models_save_path = os.path.join(checkpoint_path, f"{seed}")
        
        acc_results_seed = []
        f1_results_seed = []
        auc_results_seed = []
        for fold in range(1, num_folds + 1): # only evaluate the finetuned model here
            print(f"Evaluating seed {seed}, fold {fold}...")
            
            path, path_name = get_path_loader_new(input_path, dataset_name)
            _, _, test_idx = get_idx(parser, path, dataset_name)
            print("test_idx", test_idx)
            test_data, test_label = get_test_loader_batch(path_name, test_idx)
            print("test_data shape:", test_data.shape)
            # test_data = np.transpose(test_data, (0, 2, 1))
            TestSet = FTDataSet(test_data, test_label, dataset_name, multi_label=False)
            
            weights = torch.load(os.path.join(models_save_path, f'Finetuned_Model_{fold}.pkl'), map_location=device)
            # model.load_state_dict(weights)
            model.load_state_dict(weights, strict=False)  # in case some keys are missing

            auc, F1, acc = test(model, TestSet, batch_size=100, multi_label=False, device=parser["device"], n_classes=n_classes)

            print("Seed: {}, Fold: {}, Test Auc: {:.4f}, Test Acc: {:.4f}, Test MF1: {:.4f}".format(seed, fold, auc, acc, F1))
            acc_results_seed.append(acc)
            f1_results_seed.append(F1)
            auc_results_seed.append(auc)
            parser["Fold"] += 1
        acc_results.append(np.mean(acc_results_seed))
        f1_results.append(np.mean(f1_results_seed))
        auc_results.append(np.mean(auc_results_seed))
        print(f"Seed {seed} average results: Acc {np.mean(acc_results_seed):.4f}, MF1 {np.mean(f1_results_seed):.4f}, AUC {np.mean(auc_results_seed):.4f}")
        parser["Fold"] = 1
    
    # Compute the average and variance of the results
    print(f"All seeds average results: Acc {np.mean(acc_results):.4f} ± {np.std(acc_results):.4f}, MF1 {np.mean(f1_results):.4f} ± {np.std(f1_results):.4f}, AUC {np.mean(auc_results):.4f} ± {np.std(auc_results):.4f}")


if __name__ == '__main__':
    main()





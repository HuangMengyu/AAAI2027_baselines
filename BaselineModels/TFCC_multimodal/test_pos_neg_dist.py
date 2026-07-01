from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from cuml.manifold import TSNE
import plotly.express as px
from common_utils.dataset_loader import get_path_loader_new, get_idx, get_loader
import numpy as np
import torch
from model import Model as tfcc_model
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from base_models import FTDataSet, SSLDataSet
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
from torch.utils.data import Dataset
from base_models import channel_separation
from dataloader.augmentations import DataTransform_time
import torch.nn.functional as F
import plotly.graph_objects as go
from scipy.stats import norm

def get_test_data(data_path, fold):
    path, path_name = get_path_loader_new(data_path)

    if fold == '1':
        test_idx = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    elif fold == '2':
        test_idx = ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
    elif fold == '3':
        test_idx = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '40', '41', '42', '43', '44', '45', '46']
    elif fold == '4':
        test_idx = ['47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62']
    elif fold == '5':
        test_idx = ['63', '64', '65', '66', '67', '70', '71', '72', '73', '74', '75', '76', '77', '80', '81', '82']

    
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

    return test_data, test_label
   

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



class SSLDataSet_with_label(Dataset):
    def __init__(self, data, label):
        super(SSLDataSet_with_label, self).__init__()
        self.time = data
        self.label = label

    def __getitem__(self, idx):
        time_temp = self.time[idx]
        label = self.label[idx]
        EEG_temp, EOG_temp = channel_separation(time_temp)
        EEG_temp = EEG_temp[np.newaxis, ...]
        EOG_temp = EOG_temp[np.newaxis, ...]
        EEG_aug_temp = DataTransform_time(EEG_temp)
        EOG_aug_temp = DataTransform_time(EOG_temp)

        EEG_temp = np.squeeze(EEG_temp, axis=0)
        EOG_temp = np.squeeze(EOG_temp, axis=0)
        EEG_aug_temp = np.squeeze(EEG_aug_temp, axis=0)
        EOG_aug_temp = np.squeeze(EOG_aug_temp, axis=0)

        EEG = torch.tensor(EEG_temp, dtype=torch.float)
        EOG = torch.tensor(EOG_temp, dtype=torch.float)
        EEG_aug = torch.tensor(EEG_aug_temp, dtype=torch.float)
        EOG_aug = torch.tensor(EOG_aug_temp, dtype=torch.float)
        
        sample = (EEG, EOG, EEG_aug, EOG_aug, label)

        return sample

    def __len__(self):
        return self.time.shape[0]

def analyze_similarity_vs_class_px(experiment_log, img_save_path, features_1, features_2, labels, calc_limit=35000, plot_limit=35000):
    """
    Analyzes cosine similarities using Plotly Express (px).
    
    Args:
        features_1: (N, D) numpy array
        features_2: (N, D) numpy array
        labels: (N,) numpy array
        calc_limit: Max samples for O(N^2) matrix calculation.
        plot_limit: Max samples per group for visualization.
    """
    
    # --- Step 1: Safety Sampling (Same as before) ---
    N = len(features_1)
    if N > calc_limit:
        print(f"Dataset too large ({N}). Sampling {calc_limit} for calculation...")
        indices = np.random.choice(N, calc_limit, replace=False)
        features_1 = features_1[indices]
        features_2 = features_2[indices]
        labels = labels[indices]

    print("Computing similarity matrix...")

    
    # --- Step 2: Compute Similarities ---
    features_1 = F.normalize(features_1, dim=1)
    features_2 = F.normalize(features_2, dim=1)
    sim_matrix = torch.mm(features_1, features_2.t())
    neg_mask = ~torch.eye(sim_matrix.shape[0], dtype=bool, device=sim_matrix.device)  # all negatives
    
    # filter out true postives from the set of negative samples
    label_mask = (labels[:, None] == labels[None, :])
    true_pos_mask = label_mask & neg_mask  # only consider negatives
    print("num of true_pos", torch.sum(true_pos_mask))

    # filter out hard negatives from the set of negative samples
    hard_neg_mask = (~label_mask) & neg_mask
    print("num of hard_neg", torch.sum(hard_neg_mask))

    # Extract raw arrays
    pos_arr = torch.diag(sim_matrix).cpu().numpy()
    # pos_arr = torch.sum(features_1 * features_2, dim=-1).cpu().numpy()
    print("pos_arr", pos_arr.shape)
    print("pos_arr", pos_arr[:100])
    sim_matrix = sim_matrix.fill_diagonal_(0)
    neg_arr = torch.mean(sim_matrix, dim=-1)
    # neg_arr = sim_matrix[neg_mask]
    # print("neg_arr", neg_arr.shape)
    # print("neg_arr", neg_arr[:100])
    neg_arr = neg_arr.cpu().numpy()
    print("neg_arr", neg_arr.shape)
    true_pos_arr = sim_matrix[true_pos_mask]
    print("true_pos_arr", true_pos_arr.shape)
    print("true_pos_arr", true_pos_arr[:100])
    true_pos_arr = true_pos_arr.cpu().numpy()
    print("true_pos_arr", true_pos_arr.shape)
    hard_neg_arr = sim_matrix[hard_neg_mask]
    print("hard_neg_arr", hard_neg_arr.shape)
    print("hard_neg_arr", hard_neg_arr[:100])
    hard_neg_arr = hard_neg_arr.cpu().numpy()
    print("hard_neg_arr", hard_neg_arr.shape)

    # # put in cpu

    # print statistics
    pos_mean = pos_arr.mean()
    pos_std = pos_arr.std()
    pos_variance = pos_arr.var() # 方差是标准差的平方
    
    print("-" * 30)
    print(f"Positive pair Statistics:")
    print(f"  Mean (μ): {pos_mean:.4f}")
    print(f"  Standard Deviation (σ): {pos_std:.4f}")
    print(f"  Variance (σ²): {pos_variance:.4f}")
    
    neg_mean = neg_arr.mean()
    neg_std = neg_arr.std()
    neg_variance = neg_arr.var()
    
    print("-" * 30)
    print(f"Negative pair Statistics:")
    print(f"  Mean (μ): {neg_mean:.4f}")
    print(f"  Standard Deviation (σ): {neg_std:.4f}")
    print(f"  Variance (σ²): {neg_variance:.4f}")
    print("-" * 30)

    true_pos_mean = true_pos_arr.mean()
    true_pos_std = true_pos_arr.std()
    true_pos_variance = true_pos_arr.var() # 方差是标准差的平方
    
    print("-" * 30)
    print(f"True Positive pairs within negative pairs Statistics:")
    print(f"  Mean (μ): {true_pos_mean:.4f}")
    print(f"  Standard Deviation (σ): {true_pos_std:.4f}")
    print(f"  Variance (σ²): {true_pos_variance:.4f}")

    hard_neg_mean = hard_neg_arr.mean()
    hard_neg_std = hard_neg_arr.std()
    hard_neg_variance = hard_neg_arr.var()
    
    print("-" * 30)
    print(f"Hard Negative pair Statistics:")
    print(f"  Mean (μ): {hard_neg_mean:.4f}")
    print(f"  Standard Deviation (σ): {hard_neg_std:.4f}")
    print(f"  Variance (σ²): {hard_neg_variance:.4f}")
    print("-" * 30)

    stats = {
        'Positive pairs': {
            'mean': pos_mean,
            'std': pos_std,
        },
        'Negative pairs': {
            'mean': neg_mean,
            'std': neg_std,
        },
        'Same-Class Negatives': {
            'mean': true_pos_mean,
            'std': true_pos_std,
        },
        'Hard Negatives': {
            'mean': hard_neg_mean,
            'std': hard_neg_std,
        },

    }


    # --- Step 4: Visualization with px ---
    print("Generating Plotly Express chart...")

    # 2. Define Plotting Parameters
    x_range = np.linspace(-1, 1, 500) # Cosine similarity range
    
    # Define distinct colors for clarity
    color_map = {
        'Positive pairs': 'black',
        'Negative pairs': 'purple',
        'Same-Class Negatives': 'green',
        'Hard Negatives': 'red'
    }

    # 3. Create Plotly Figure
    fig = go.Figure()

    for pair_type, data in stats.items():
        mu = data['mean']
        sigma = data['std']

        # Calculate the Gaussian Probability Density Function (PDF) values
        pdf = norm.pdf(x_range, loc=mu, scale=sigma)

        # Add the smooth PDF curve to the figure
        fig.add_trace(go.Scatter(
            x=x_range,
            y=pdf,
            mode='lines',
            name=f"{pair_type} (Mean={mu:.2f})",
            line=dict(color=color_map[pair_type], width=3),
            hovertemplate=f"Mean: {mu:.4f}<br>Std: {sigma:.4f}<br>Similarity: %{{x:.2f}}<br>Density: %{{y:.2f}}<extra></extra>"
        ))
        
        # Add a vertical line to explicitly mark the mean
        fig.add_vline(x=mu, 
                      line_dash="dot", 
                      line_color=color_map[pair_type],
                      annotation_text=f"Mean: {mu:.2f}",
                      annotation_position="top right",
                      annotation_font=dict(color=color_map[pair_type], size=10))
    

    fig.update_layout(
        title=f"<b>Cosine Similarities_{experiment_log}</b>",
        xaxis_title="Cosine Similarity",
        yaxis_title="Probability Density",
        template="plotly_white",
        width=900,
        height=600,
        legend_title_text="Relationship"
    )

    # fig.show()
    img_name = f"Cosinesimilarity_distribution_{experiment_log}.png"
    img_save_path = os.path.join(img_save_dir, img_name)
    try:
        fig.write_image(img_save_path)
        print("Plot saved to:", img_save_path)
    except Exception as e:
        print(f"Failed to save image (check Kaleido setup/memory): {e}")



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

def test(model, dataset, batch_size, multi_label):
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
            EEG, EOG, y = tuple(t.to(device) for t in batch)
            x = (EEG, EOG)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])
            # label.extend([y.cpu().detach().numpy().tolist()])
            # num += 1
            # if num == 1:
            #     break

    # label = np.array(label)
    # label = np.reshape(label, (-1,))
    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes=5)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--raw_input", type=str2bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette")
    parser.add_argument("--experiment_log", type=str, default="No log")
    parser.add_argument("--img_save_dir", type=str, default="No log")
    parser.add_argument("--fold", type=str, default="1")
    parser.add_argument("--real_label", type=str2bool, default="False")
    parser.add_argument("--intra_modal", type=str2bool, default=False) # if true, do intra modal comparisoon
    parser.add_argument("--inter_modal", type=str2bool, default=False) # if true, do cross modal comparison
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # raw_input = opt.raw_input
    checkpoint_path = opt.checkpoint_path
    data_path = opt.data_path
    experiment_log = opt.experiment_log
    img_save_dir = opt.img_save_dir
    fold = opt.fold
    real_label = opt.real_label

    # print("raw_input:", raw_input)
    print("checkpoint_path:", checkpoint_path)
    print("data_path:", data_path)
    print("experiment_log:", experiment_log)
    print("img_save_dir:", img_save_dir)
    print("fold:", fold)
    print("real_label:", real_label)
    
    inter_modal = opt.inter_modal
    intra_modal = opt.intra_modal
    if inter_modal:
        print("inter_modal")
    elif intra_modal:
        print("intra_modal")
    if inter_modal and intra_modal:
        print("Both inter_modal and intra_modal are set to True. Defaulting to inter_modal only.")
        intra_modal = False

    test_data, test_label = get_test_data(data_path, fold)

    # img_name = f"{experiment_log}.png"
    # img_save_path = os.path.join(img_save_dir, img_name)

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)

    
    model = tfcc_model(
        output_dims=127,
        device=device,
        num_class=5,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        timesteps=50, 
        hidden_dim=64,
    )
    weight_load_path = os.path.join(checkpoint_path, f'Finetuned_Model_{fold}.pkl')
    # weight_load_path = os.path.join(checkpoint_path, f'Pretrained_Model_1.pkl')
    pretrained_weights = torch.load(weight_load_path, map_location=device)
    model.load_state_dict(pretrained_weights)
    model.to(device)
    model.eval()
    
    # test the data first
    TestSet = FTDataSet(test_data, test_label, multi_label=False)
    auc1, F11, acc1, _ = test(model, TestSet, batch_size=100, multi_label=False)
    
    
    
    
    
    if inter_modal:
        test_data = FTDataSet(test_data, test_label, multi_label=False)
        testloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    elif intra_modal:
        test_data = SSLDataSet_with_label(test_data, test_label)
        testloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    


    EEG_feat = []
    EOG_feat = []
    
    if intra_modal:
        EEG_aug_feat = []    
        EOG_aug_feat = []    
    
    labels = []
    with torch.no_grad():
        for batch in tqdm(testloader):
            if inter_modal:
                EEG, EOG, y = tuple(t.to(device) for t in batch)

                EEG_features = model.EEG_encoder(EEG)
                EOG_features = model.EOG_encoder(EOG)

                EEG_features = F.normalize(EEG_features, dim=1)
                EOG_features = F.normalize(EOG_features, dim=1)
                EEG_features_flat = model.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2)).reshape(EEG.shape[0], -1)
                EOG_features_flat = model.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2)).reshape(EOG.shape[0], -1)

            elif intra_modal:
                EEG, EOG, EEG_aug, EOG_aug, y = tuple(t.to(device) for t in batch)

                EEG_features = model.EEG_encoder(EEG)
                EOG_features = model.EOG_encoder(EOG)
                EEG_aug_features = model.EEG_encoder(EEG_aug)
                EOG_aug_features = model.EOG_encoder(EOG_aug)

                EEG_features = F.normalize(EEG_features, dim=1)
                EOG_features = F.normalize(EOG_features, dim=1)
                EEG_aug_features = F.normalize(EEG_aug_features, dim=1)
                EOG_aug_features = F.normalize(EOG_aug_features, dim=1)
                EEG_features_flat = model.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2)).reshape(EEG.shape[0], -1)
                EOG_features_flat = model.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2)).reshape(EOG.shape[0], -1)
                EEG_aug_features_flat = model.TFCC.EEG_contrasting.seq_transformer(EEG_aug_features.transpose(1,2)).reshape(EEG.shape[0], -1)
                EOG_aug_features_flat = model.TFCC.EOG_contrasting.seq_transformer(EOG_aug_features.transpose(1,2)).reshape(EOG.shape[0], -1)


            EEG_feat.append(EEG_features_flat)
            EOG_feat.append(EOG_features_flat)

            if intra_modal:
                EEG_aug_feat.append(EEG_aug_features_flat)
                EOG_aug_feat.append(EOG_aug_features_flat)


            if real_label:
                labels_temp = y
            else:
                features_flat = torch.cat((EEG_features_flat, EOG_features_flat), dim=1)
                labels_temp = model.classifier(features_flat).argmax(dim=1)


            # labels_temp = y.cpu().numpy()
            # labels_temp = y
            labels.append(labels_temp)

    EEG_features_flat = torch.cat(EEG_feat, axis=0)
    EOG_features_flat = torch.cat(EOG_feat, axis=0)
    if intra_modal:
        EEG_aug_features_flat = torch.cat(EEG_aug_feat, axis=0)
        EOG_aug_features_flat = torch.cat(EOG_aug_feat, axis=0)
    
    # labels_flat = np.concatenate(labels, axis=0)
    labels_flat = torch.cat(labels, axis=0)
    
    if inter_modal:
        analyze_similarity_vs_class_px(experiment_log + "_inter_modal", img_save_dir, EEG_features_flat, EOG_features_flat, labels_flat)
    elif intra_modal:
        analyze_similarity_vs_class_px(experiment_log + "_EEG_aug", img_save_dir, EEG_features_flat, EEG_aug_features_flat, labels_flat)
        analyze_similarity_vs_class_px(experiment_log + "_EOG_aug", img_save_dir, EOG_features_flat, EOG_aug_features_flat, labels_flat)
    
    



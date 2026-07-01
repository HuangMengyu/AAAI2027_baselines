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
from base_models import FTDataSet

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
# from einops import rearrange, repeat

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


def validate_hypothesis_roc(experiment_log, img_save_dir, features, labels, sample_limit=35000):
    """
    Validates if Cosine Similarity can distinguish 'Same Class' from 'Diff Class'
    using ROC-AUC analysis.
    """
    
    # 1. Safety Sampling (Avoid O(N^2) memory crash)
    N = len(features)
    if N > sample_limit:
        print(f"Sampling {sample_limit} data points for validation...")
        indices = np.random.choice(N, sample_limit, replace=False)
        features = features[indices]
        labels = labels[indices]

    print("Computing similarity pairs...")
    
    # 2. Compute Similarity Matrix
    sim_matrix = cosine_similarity(features)
    
    # 3. Create Ground Truth Matrix (1 = Same Class, 0 = Diff Class)
    label_matrix = (labels[:, None] == labels[None, :]).astype(int)
    
    # 4. Flatten and Remove Diagonal/Duplicates
    # We only care about the upper triangle (excluding diagonal self-matches)
    mask = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)
    
    y_true = label_matrix[mask]  # The actual "Is Same Class?" (0 or 1)
    y_scores = sim_matrix[mask]  # The Cosine Similarity (-1 to 1)

    # 5. Calculate ROC Metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"------------------------------------------------")
    print(f"Validation Result (ROC-AUC Score): {roc_auc:.4f}")
    print(f"------------------------------------------------")
    
    if roc_auc > 0.9:
        print("CONCLUSION: Hypothesis STRONGLY VALIDATED.")
    elif roc_auc > 0.7:
        print("CONCLUSION: Hypothesis WEAKLY VALIDATED (Some overlap exists).")
    else:
        print("CONCLUSION: Hypothesis REJECTED (Similarity does not imply class).")

    # 6. Plot ROC Curve with Plotly Express
    # We downsample the curve for plotting speed if it has too many points
    if len(fpr) > 5000000:
        indices = np.linspace(0, len(fpr)-1, 5000).astype(int)
        fpr_plot, tpr_plot = fpr[indices], tpr[indices]
    else:
        fpr_plot, tpr_plot = fpr, tpr

    df_roc = pd.DataFrame({
        'False Positive Rate': fpr_plot, 
        'True Positive Rate': tpr_plot
    })
    
    fig = px.area(
        df_roc, 
        x='False Positive Rate', 
        y='True Positive Rate',
        title=f'<b>Hypothesis Validation ROC Curve</b><br>AUC Score: {roc_auc:.4f} Experiment: {experiment_log}',
        labels={'False Positive Rate': 'Fraction of Diff-Class Pairs falsely marked similar', 
                'True Positive Rate': 'Fraction of Same-Class Pairs correctly marked similar'},
        width=700, height=600,
        template='plotly_white'
    )
    
    # Add diagonal reference line (Random Chance)
    fig.add_shape(
        type='line', line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.show()
    img_name = f"AUC-ROC_{experiment_log}.png"
    img_save_path = os.path.join(img_save_dir, img_name)
    fig.write_image(img_save_path)
    print("Plot saved to:", img_save_path)

# Usage
# validate_hypothesis_roc(features_flat, labels_flat)

def analyze_similarity_vs_class_px(experiment_log, img_save_path, features, labels, calc_limit=35000, plot_limit=35000):
    """
    Analyzes cosine similarities using Plotly Express (px).
    
    Args:
        features: (N, D) numpy array
        labels: (N,) numpy array
        calc_limit: Max samples for O(N^2) matrix calculation.
        plot_limit: Max samples per group for visualization.
    """
    
    # --- Step 1: Safety Sampling (Same as before) ---
    N = len(features)
    if N > calc_limit:
        print(f"Dataset too large ({N}). Sampling {calc_limit} for calculation...")
        indices = np.random.choice(N, calc_limit, replace=False)
        features_sub = features[indices]
        labels_sub = labels[indices]
    else:
        features_sub = features
        labels_sub = labels

    print("Computing similarity matrix...")
    
    # --- Step 2: Compute Similarities ---
    sim_matrix = cosine_similarity(features_sub)
    label_matrix = labels_sub[:, None] == labels_sub[None, :]
    mask_upper = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)

    # Extract raw arrays
    intra_arr = sim_matrix[label_matrix & mask_upper]
    inter_arr = sim_matrix[(~label_matrix) & mask_upper]

    intra_mean = intra_arr.mean()
    intra_std = intra_arr.std()
    intra_variance = intra_arr.var() # 方差是标准差的平方
    
    print("-" * 30)
    print(f"Same Class (Intra) Statistics:")
    print(f"  Mean (μ): {intra_mean:.4f}")
    print(f"  Standard Deviation (σ): {intra_std:.4f}")
    print(f"  Variance (σ²): {intra_variance:.4f}")
    
    # 打印 不同类别 (Inter) 的统计量
    inter_mean = inter_arr.mean()
    inter_std = inter_arr.std()
    inter_variance = inter_arr.var()
    
    print("-" * 30)
    print(f"Different Class (Inter) Statistics:")
    print(f"  Mean (μ): {inter_mean:.4f}")
    print(f"  Standard Deviation (σ): {inter_std:.4f}")
    print(f"  Variance (σ²): {inter_variance:.4f}")
    print("-" * 30)

    # exit(0)

    # print(f"Avg Intra-class: {intra_arr.mean():.4f}")
    # print(f"Avg Inter-class: {inter_arr.mean():.4f}")

    # --- Step 3: Prepare DataFrame for Plotly Express ---
    # Plotly Express loves "Long Form" DataFrames.
    # We create a DF with columns: [Similarity, Type]
    
    def safe_sample(data, limit):
        if len(data) > limit:
            return np.random.choice(data, limit, replace=False)
        return data

    # Sample and label the data
    s_intra = safe_sample(intra_arr, plot_limit)
    s_inter = safe_sample(inter_arr, plot_limit)

    # Construct DataFrame
    df_intra = pd.DataFrame({'Similarity': s_intra, 'Type': 'Same Class (Intra)'})
    df_inter = pd.DataFrame({'Similarity': s_inter, 'Type': 'Different Class (Inter)'})
    
    df_plot = pd.concat([df_intra, df_inter], axis=0, ignore_index=True)

    # --- Step 4: Visualization with px ---
    print("Generating Plotly Express chart...")
    
    fig = px.histogram(
        df_plot, 
        x="Similarity", 
        color="Type", 
        nbins=100,              # Granularity of the bins
        barmode="overlay",      # Allows comparison without stacking
        histnorm='probability density', # Normalizes so area = 1 (fair comparison)
        opacity=0.6,            # See-through bars
        marginal="box",         # Adds a mini box-plot above to see median/outliers
        color_discrete_map={    # Custom colors
            'Same Class (Intra)': '#2ca02c', # Green
            'Different Class (Inter)': '#d62728' # Red
        },
        range_x=[0, 1]         # Fix x-axis to cosine range
    )

    fig.update_layout(
        title=f"<b>Distribution of Latent Cosine Similarities_{experiment_log}</b>",
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
    fig.write_image(img_save_path)
    print("Plot saved to:", img_save_path)

# Usage
# analyze_similarity_vs_class_px(features_flat, labels_flat)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--raw_input", type=str2bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette")
    parser.add_argument("--experiment_log", type=str, default="No log")
    parser.add_argument("--img_save_dir", type=str, default="No log")
    parser.add_argument("--fold", type=str, default="1")
    parser.add_argument("--real_label", type=str2bool, default="False")
    parser.add_argument("--use_transformer", type=str2bool, default="False")
    parser.add_argument("--use_projector", type=str2bool, default=False)
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # raw_input = opt.raw_input
    checkpoint_path = opt.checkpoint_path
    data_path = opt.data_path
    experiment_log = opt.experiment_log
    img_save_dir = opt.img_save_dir
    fold = opt.fold
    real_label = opt.real_label

    use_transformer = opt.use_transformer
    use_projector = opt.use_projector

    # print("raw_input:", raw_input)
    print("checkpoint_path:", checkpoint_path)
    print("data_path:", data_path)
    print("experiment_log:", experiment_log)
    print("img_save_dir:", img_save_dir)
    print("fold:", fold)
    print("real_label:", real_label)

    if use_transformer:
        print("Using Transformer features as finetuning input features.")
    if use_projector:
        print("Using Projector features as finetuning input features.")
    if use_transformer and use_projector:
        print("Using Projector features as finetuning input features.")
        use_transformer = False  # If using projector, we do not use transformer features.


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
        use_transformer = use_transformer,
        use_projector = use_projector,
    )
    weight_load_path = os.path.join(checkpoint_path, f'Finetuned_Model_{fold}.pkl')
    # weight_load_path = os.path.join(checkpoint_path, f'Pretrained_Model_1.pkl')
    pretrained_weights = torch.load(weight_load_path, map_location=device)
    model.load_state_dict(pretrained_weights)
    model.to(device)
    

    test_data = FTDataSet(test_data, test_label, multi_label=False)
    testloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)

    feat = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(testloader):
            # batch = batch.float().to(device)
            EEG, EOG, y = tuple(t.to(device) for t in batch)

            EEG_features = model.EEG_encoder(EEG)
            EOG_features = model.EOG_encoder(EOG)

            if use_transformer:
                EEG_transformer_features = model.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2)).reshape(EEG.shape[0], -1)
                EOG_transformer_features = model.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2)).reshape(EOG.shape[0], -1)
                features_flat = torch.cat((EEG_transformer_features, EOG_transformer_features), dim=1)
            elif use_projector:
                EEG_transformer_features = model.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2))
                EOG_transformer_features = model.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2))
                EEG_features = model.TFCC.EEG_contrasting.projection_head(EEG_transformer_features)
                EOG_features = model.TFCC.EOG_contrasting.projection_head(EOG_transformer_features)
                EEG_features_flat = EEG_features.reshape(EEG.shape[0], -1)
                EOG_features_flat = EOG_features.reshape(EOG.shape[0], -1)
                features_flat = torch.cat((EEG_features_flat, EOG_features_flat), dim=1)
            else:
                EEG_features_flat = EEG_features.reshape(EEG.shape[0], -1)
            
                EOG_features_flat = EOG_Features.reshape(EOG.shape[0], -1)

                features_flat = torch.cat((EEG_features_flat, EOG_features_flat), dim=1)

            if real_label:
                labels_temp = y.cpu().numpy()
            else:
                labels_temp = model.classifier(features_flat).argmax(dim=1).cpu().numpy()

            features_flat = features_flat.cpu().numpy()

            feat.append(features_flat)
            labels.append(labels_temp)
    features_flat = np.concatenate(feat, axis=0)
    labels_flat = np.concatenate(labels, axis=0)
    
    analyze_similarity_vs_class_px(experiment_log, img_save_dir, features_flat, labels_flat)
    validate_hypothesis_roc(experiment_log, img_save_dir, features_flat, labels_flat)
    




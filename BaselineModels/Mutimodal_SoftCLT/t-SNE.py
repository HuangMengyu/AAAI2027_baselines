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
   

def TSNE_fit(input):
    # tsne = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate=50)
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(input)
    print("Finished PCA preprocessing")

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    print("Finished t-SNE fitting.")
    return X_tsne

def plot_tsne(X_tsne, y, img_save_path="No log"):
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)
    fig.update_layout(
        title=f"t-SNE visualization of {experiment_log}",
        # xaxis_title="First t-SNE",
        # yaxis_title="Second t-SNE",
    )
    # fig.show()
    fig.write_image(img_save_path)
    print("t-SNE plot saved to:", img_save_path)

def load_pretrained_encoders(new_model, checkpoint_path, device):
    """
    Loads weights for encoders and TFCC, but skips the classifier.
    """
    # 1. Load the checkpoint from the file
    # Note: Adjust map_location if you are loading on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle cases where checkpoint is a dict with metadata (e.g. {'state_dict': ..., 'epoch': ...})
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # 2. Get the state_dict of the NEW model
    # We need this to ensure we only update keys that actually exist and match
    model_dict = new_model.state_dict()

    # 3. Filter the PRETRAINED checkpoint
    # Logic: Keep the key IF:
    #   a. It is NOT part of the 'classifier'
    #   b. It exists in the new model (safety check)
    #   c. The shape matches the new model (safety check)
    pretrained_dict = {
        k: v for k, v in checkpoint.items() 
        if 'classifier' not in k 
        and k in model_dict 
        and v.shape == model_dict[k].shape
    }

    # 4. Verification Print (Optional but recommended)
    print(f"Total keys in checkpoint: {len(checkpoint)}")
    print(f"Keys matching and loading: {len(pretrained_dict)}")
    print(f"Keys skipped (Classifier or mismatch): {len(checkpoint) - len(pretrained_dict)}")

    # 5. Update the new model's dictionary with the pretrained weights
    model_dict.update(pretrained_dict)

    # 6. Load the updated dictionary into the model
    # We use strict=True here because we manually handled the matching above.
    # This ensures that 'model_dict' is now a perfect valid state for new_model.
    new_model.load_state_dict(model_dict)
    
    return new_model



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input", type=str2bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette")
    parser.add_argument("--experiment_log", type=str, default="No log")
    parser.add_argument("--img_save_dir", type=str, default="No log")
    parser.add_argument("--fold", type=str, default="1")
    parser.add_argument("--use_transformer", type=str2bool, default=False)
    parser.add_argument("--use_projector", type=str2bool, default=False)
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_input = opt.raw_input
    checkpoint_path = opt.checkpoint_path
    data_path = opt.data_path
    experiment_log = opt.experiment_log
    img_save_dir = opt.img_save_dir
    fold = opt.fold
    use_transformer = opt.use_transformer
    use_projector = opt.use_projector
    

    print("raw_input:", raw_input)
    print("checkpoint_path:", checkpoint_path)
    print("data_path:", data_path)
    print("experiment_log:", experiment_log)
    print("img_save_dir:", img_save_dir)
    print("fold:", fold)
    if use_transformer:
        print("Using Transformer features as finetuning input features.")
    if use_projector:
        print("Using Projector features as finetuning input features.")
    if use_transformer and use_projector:
        print("Using Projector features as finetuning input features.")
        use_transformer = False  # If using projector, we do not use transformer features.


    test_data, test_label = get_test_data(data_path, fold)

    img_name = f"{experiment_log}.png"
    img_save_path = os.path.join(img_save_dir, img_name)

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)

    if raw_input: # just map the raw input
        # test_data = test_data[:10000]
        # print("Take only first 10000 samples ")
        test_data_input = np.reshape(test_data, (test_data.shape[0], -1))
        print("test_data_input shape:", test_data_input.shape)
        X_tsne = TSNE_fit(test_data_input)
        plot_tsne(X_tsne, test_label, img_save_path)
    else:
        feat = []
        model = tfcc_model(
            output_dims=127,
            device=device,
            num_class=5,
            final_out_channels = 128, 
            kernel_size = 25, 
            stride = 3, 
            timesteps=50, 
            hidden_dim=64,
            use_transformer=use_transformer,
            use_projector=use_projector,
        )
        weight_load_path = os.path.join(checkpoint_path, f'Finetuned_Model_{fold}.pkl')
        # weight_load_path = os.path.join(checkpoint_path, f'Pretrained_Model_1.pkl')
        if use_transformer or use_projector:
            model = load_pretrained_encoders(model, weight_load_path, device)
        else:
            model.load_state_dict(torch.load(weight_load_path))
        model.to(device)

        

        # model = torch.load(os.path.join(checkpoint_path, f'Finetuned_Model_1.pkl'), map_location=device)
        # test_data = test_data[:1000]
        # print("Take only first 10000 samples ")
        test_data = FTDataSet(test_data, test_label, multi_label=False)
        testloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
        # tensor_data = torch.from_numpy(test_data).float().to(device)
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
                features_flat = features_flat.cpu().numpy()
                feat.append(features_flat)

                labels_temp = y.cpu().numpy()
                labels.append(labels_temp)
        features_flat = np.concatenate(feat, axis=0)
        labels_flat = np.concatenate(labels, axis=0)
        if np.any(np.isnan(features_flat)):
            print("CRITICAL ERROR: Features contain NaNs!")
            # Option A: Replace NaNs with 0 (safe for visualization)
            features_flat = np.nan_to_num(features_flat)
            
        # Check for Infinity
        if np.any(np.isinf(features_flat)):
            print("CRITICAL ERROR: Features contain Infinity!")
            # Option A: Replace Inf with large finite numbers
            features_flat = np.nan_to_num(features_flat)
        # print("features_flat shape:", features_flat.shape)
        X_tsne = TSNE_fit(features_flat)
        plot_tsne(X_tsne, labels_flat, img_save_path)

    




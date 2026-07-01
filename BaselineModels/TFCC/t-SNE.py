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
# from einops import rearrange, repeat

def get_test_data(data_path):
    path, path_name = get_path_loader_new(data_path)

    # ONLY TEST FOLD 1
    test_idx = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
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
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_input = opt.raw_input
    checkpoint_path = opt.checkpoint_path
    data_path = opt.data_path
    experiment_log = opt.experiment_log
    img_save_dir = opt.img_save_dir

    print("raw_input:", raw_input)
    print("checkpoint_path:", checkpoint_path)
    print("data_path:", data_path)
    print("experiment_log:", experiment_log)
    print("img_save_dir:", img_save_dir)

    test_data, test_label = get_test_data(data_path)

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
        # model = tfcc_model(
        #     input_dims=3,
        #     output_dims=127,
        #     device=device,
        #     num_class=5,
        #     final_out_channels = 128, 
        #     kernel_size = 25, 
        #     stride = 3, 
        #     timesteps=50, 
        #     hidden_dim=64,
        # )
        
        feat = []
        model = torch.load(os.path.join(checkpoint_path, f'Finetuned_Model_1.pkl'), map_location=device)
        # test_data = test_data[:1000]
        # print("Take only first 10000 samples ")
        testloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
        # tensor_data = torch.from_numpy(test_data).float().to(device)
        with torch.no_grad():
            for batch in tqdm(testloader):
                batch = batch.float().to(device)
                features = model.encoder(batch)
                features_flat = features.reshape(features.shape[0], -1)
                # print("features_flat shape:", features_flat.shape)
                feat.append(features_flat)
        features_flat = torch.cat(feat, dim=0).cpu().numpy()
        # print("features_flat shape:", features_flat.shape)
        X_tsne = TSNE_fit(features_flat)
        plot_tsne(X_tsne, test_label, img_save_path)

    




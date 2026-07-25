import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


DATASET_CLASS_COUNTS = {
    "SleepEDFx": 5,
    "PAMAP2": 9,
    "UCI-HAR": 6,
    "UCI-HAR_total": 6,
}


def cluster_counts_for_dataset(dataset_name, multipliers):
    if dataset_name not in DATASET_CLASS_COUNTS:
        raise ValueError(f"Unsupported IFND dataset_name: {dataset_name}")
    base_count = DATASET_CLASS_COUNTS[dataset_name]
    return [max(2, int(base_count * multiplier)) for multiplier in multipliers]


class IFNDPseudoLabeler:
    def __init__(
        self,
        dataset_name,
        cluster_multipliers=(1, 3, 10),
        final_acceptance=1.0,
        cluster_every=1,
        random_state=0,
    ):
        self.dataset_name = dataset_name
        self.cluster_counts = cluster_counts_for_dataset(dataset_name, cluster_multipliers)
        self.final_acceptance = float(final_acceptance)
        self.cluster_every = max(1, int(cluster_every))
        self.random_state = int(random_state)
        self.pseudo_labels = None
        self.last_epoch = None
        self.last_stats = {}

    def should_update(self, epoch):
        return self.pseudo_labels is None or epoch % self.cluster_every == 0

    def batch_labels(self, index, device):
        if self.pseudo_labels is None:
            return None
        index = index.detach().cpu().long()
        return self.pseudo_labels[:, index].to(device)

    def update(self, tfcc_model, train_loader, epoch, total_epochs):
        features = self.extract_features(tfcc_model, train_loader)
        acceptance_rate = self.acceptance_rate(epoch, total_epochs)
        pseudo_labels = []
        stats = []
        num_samples = features.shape[0]

        for cluster_count in self.cluster_counts:
            effective_clusters = min(cluster_count, num_samples)
            labels = np.full(num_samples, -1, dtype=np.int64)
            if effective_clusters < 2:
                pseudo_labels.append(labels)
                stats.append({
                    "clusters": effective_clusters,
                    "accepted": 0,
                    "total": num_samples,
                })
                continue

            kmeans = KMeans(
                n_clusters=effective_clusters,
                n_init=10,
                random_state=self.random_state + epoch + effective_clusters,
            )
            raw_labels = kmeans.fit_predict(features)
            confidence = self.confidence(features, kmeans.cluster_centers_, raw_labels)
            accepted = self.accept_top_confident(confidence, acceptance_rate)
            labels[accepted] = raw_labels[accepted]
            pseudo_labels.append(labels)
            stats.append({
                "clusters": effective_clusters,
                "accepted": int(accepted.sum()),
                "total": num_samples,
            })

        self.pseudo_labels = torch.from_numpy(np.stack(pseudo_labels, axis=0)).long()
        self.last_epoch = epoch
        self.last_stats = {
            "acceptance_rate": acceptance_rate,
            "cluster_stats": stats,
        }
        return self.last_stats

    def acceptance_rate(self, epoch, total_epochs):
        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = float(epoch) / float(total_epochs - 1)
        return max(0.0, min(self.final_acceptance * progress, self.final_acceptance))

    @staticmethod
    def confidence(features, centroids, labels):
        features_t = torch.from_numpy(features).float()
        centroids_t = torch.from_numpy(centroids).float()
        features_t = F.normalize(features_t, dim=1)
        centroids_t = F.normalize(centroids_t, dim=1)
        probs = F.softmax(torch.mm(features_t, centroids_t.t()), dim=1)
        row_idx = torch.arange(features_t.shape[0])
        return probs[row_idx, torch.from_numpy(labels).long()].numpy()

    @staticmethod
    def accept_top_confident(confidence, acceptance_rate):
        accepted = np.zeros(confidence.shape[0], dtype=bool)
        num_accept = int(round(confidence.shape[0] * acceptance_rate))
        if num_accept <= 0:
            return accepted
        if num_accept >= confidence.shape[0]:
            accepted[:] = True
            return accepted
        top_idx = np.argpartition(-confidence, num_accept - 1)[:num_accept]
        accepted[top_idx] = True
        return accepted

    @staticmethod
    def extract_features(tfcc_model, train_loader):
        was_training = tfcc_model.training
        tfcc_model.eval()
        device = tfcc_model.device
        dataset_len = len(train_loader.dataset)
        output = None

        with torch.no_grad():
            for batch in train_loader:
                EEG, EOG, EEG_aug, EOG_aug, index = batch
                del EEG_aug, EOG_aug
                EEG = EEG.to(device)
                EOG = EOG.to(device)
                index = index.long()

                EEG_feat = tfcc_model.EEG_encoder(EEG)
                EOG_feat = tfcc_model.EOG_encoder(EOG)
                z_T = tfcc_model.EEG_contrasting.seq_transformer(EEG_feat.transpose(1, 2))
                z_F = tfcc_model.EOG_contrasting.seq_transformer(EOG_feat.transpose(1, 2))
                batch_features = F.normalize(torch.cat((z_T, z_F), dim=1), dim=1).cpu()

                if output is None:
                    output = torch.empty(dataset_len, batch_features.shape[1], dtype=batch_features.dtype)
                output[index] = batch_features

        if was_training:
            tfcc_model.train()
        return output.numpy()

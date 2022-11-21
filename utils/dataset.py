import pandas as pd
import torch
import os

from torch.utils.data.dataset import Dataset


class CICDDOSDataset(Dataset):
    def __init__(self, features_file, targets_file, transform=None, target_transform=None):
        self.features = pd.read_pickle(features_file)
        self.labels = pd.read_pickle(targets_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.features.iloc[index, :]
        label = self.labels.iloc[index]
        if self.transform:
            feature = self.transform(feature.values, dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label, dtype=torch.int64)
        return feature, label


def get_dataset(DATA_DIR):
    # Đọc dataset
    train_data = CICDDOSDataset(
        features_file=os.path.join(DATA_DIR, 'processed', r'train\train_features_pca.pkl'),
        targets_file=os.path.join(DATA_DIR, 'processed', r'train\train_labels.pkl'),
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    test_data = CICDDOSDataset(
        features_file=os.path.join(DATA_DIR, 'processed', r'test\test_features_pca.pkl'),
        targets_file=os.path.join(DATA_DIR, 'processed', r'test\test_labels.pkl'),
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    val_data = CICDDOSDataset(
        features_file=os.path.join(DATA_DIR, 'processed', r'val\val_features_pca.pkl'),
        targets_file=os.path.join(DATA_DIR, 'processed', r'val\val_labels.pkl'),
        transform=torch.tensor,
        target_transform=torch.tensor
    )
    return train_data, test_data, val_data


def load_data(DATA_DIR, batch_size: int):
    train_data, test_data, val_data = get_dataset(DATA_DIR)

    # Tạo data loaders cho train_data và test_data
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader

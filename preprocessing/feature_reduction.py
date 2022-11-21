import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.abspath(".."), "data")
IMG_DIR = os.path.join(os.path.abspath(".."), "images")

X_train = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features.pkl'))
X_val = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features.pkl'))
X_test = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features.pkl'))

y_train = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_labels.pkl'))
y_val = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_labels.pkl'))
y_test = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_labels.pkl'))

pca = PCA(0.99)

X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
X_val_pca = pd.DataFrame(pca.transform(X_val))
X_test_pca = pd.DataFrame(pca.transform(X_test))

X_train_pca.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features_pca.pkl'))
X_test_pca.to_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features_pca.pkl'))
X_val_pca.to_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features_pca.pkl'))

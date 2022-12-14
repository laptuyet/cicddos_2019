import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.abspath(".."), "data")
IMG_DIR = os.path.join(os.path.abspath(".."), "images")

# Data phải dc scale mới dùng PCA (data đã scale từ 81->49 features)
X_train = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features.pkl'))
X_val = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features.pkl'))
X_test = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features.pkl'))

pca = PCA(0.95)
pca.fit(X_train)
print("Cumulative Variances (Percentage):")
print(pca.explained_variance_ratio_.cumsum() * 100)

components = len(pca.explained_variance_ratio_)

# Số component sau khi dùng PCA
print(f'\nNumber of components: {components}')

plt.plot(range(1, components + 1),
         np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.savefig(os.path.join(IMG_DIR, 'pca_explained_variance.pdf'))
plt.close()

pca_components = abs(pca.components_)
print('\nTop 4 most important features in each component')
print('===============================================')
for row in range(pca_components.shape[0]):
    # get the indices of the top 4 values in each row
    temp = np.argpartition(-(pca_components[row]), 4)

    # sort the indices in descending order
    indices = temp[np.argsort((-pca_components[row])[temp])][:4]

    # print the top 4 feature names
    print(f'Component {row + 1}: {X_train.columns[indices].to_list()}')


X_train_pca = pd.DataFrame(pca.transform(X_train))
X_val_pca = pd.DataFrame(pca.transform(X_val))
X_test_pca = pd.DataFrame(pca.transform(X_test))


loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_),
    columns=[f'PC{i}' for i in range(1, len(X_train_pca.columns) + 1)],
    index=X_train.columns
)

pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('PCA loading scores (first principal component)', size=12)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'pca.pdf'))

X_train_pca.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features_pca.pkl'))
X_test_pca.to_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features_pca.pkl'))
X_val_pca.to_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features_pca.pkl'))


import pandas as pd
import os
import torch
import torch.optim as optim

from utils import (dataset, train, test, visualize)
from models.DBN import DBN
from sklearn.metrics import classification_report

DATA_DIR = os.path.join(os.path.abspath('.'), "data")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINED_MODEL_DIR = os.path.join(os.path.abspath('.'), "trained_model")

n_features = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features_pca.pkl')).shape[1]
print(f'\nNumber of components: {n_features}')

model = DBN(
    n_visible=n_features,
    n_hidden=[256, 256, 128, 128, 64],
    n_classes=6,
    learning_rate=[0.05, 0.05, 0.05, 0.05, 0.05],
    momentum=[0, 0, 0, 0, 0],
    decay=[0, 0, 0, 0, 0],
    batch_size=[128, 128, 128, 128, 128],
    num_epochs=[5, 5, 5, 5, 5],
    k=[1, 1, 1, 1, 1],
    device=DEVICE
)
# model = model.to(DEVICE)

# Load data
train_loader, test_loader, val_loader = dataset.load_data(DATA_DIR, batch_size=128)

# Pre-train DBN
# print('Start pre-training...')
# model.fit(train_loader)
# print('Finished pre-train!')

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
# optimizer = [torch.optim.Adam(
#     lr=0.05,
#     weight_decay=0,
#     amsgrad=False,
#     params=m.parameters()
# ) for m in model.models]
optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in model.models]
optimizer.append(optim.Adam(model.fc.parameters(), lr=0.001))

print('Start training...')
train_history = train.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=val_loader,
    num_epochs=30,
    device=DEVICE
)
print('Finished train!')

TRAINED_MODEL = os.path.join(TRAINED_MODEL_DIR, 'trained_DBN.pt')
TRAINED_HIS = os.path.join(TRAINED_MODEL_DIR, 'trained_history.pt')

# Save model, không cần dùng khi đã có trained_DBN.pt
# print('Saving model!')
# torch.save(model, TRAINED_MODEL)
# torch.save(train_history, TRAINED_HIS)
# print('Saving done!')

# Load model, chỉ dùng khi model đã dc trained
print('Loading model')
model = torch.load(TRAINED_MODEL)
train_history = torch.load(TRAINED_HIS)
print('Loading done')

labels = ['Benign', 'Portmap', 'DDos', 'Syn', 'TFTP', 'UDPLag']
# labels = ['Attack', 'Benign']

# Report
train_output_true = train_history['train']['output_true']
train_output_pred = train_history['train']['output_pred']

print("Training Set -- Classification Report", end="\n\n")
print(classification_report(
    y_true=train_output_true,
    y_pred=train_output_true,
    target_names=labels,
))
visualize.save_confuse_matrix(y_true=train_output_true, y_pred=train_output_pred, labels=labels)


valid_output_true = train_history['valid']['output_true']
valid_output_pred = train_history['valid']['output_pred']
print("Validation Set -- Classification Report", end="\n\n")
print(classification_report(valid_output_true, valid_output_pred, target_names=labels))


print('\n\nTesting...')
test_history = test.test(
    model=model,
    criterion=criterion,
    test_loader=test_loader,
    device=DEVICE
)
print('Finished testing!')

test_out_true = test_history['test']['output_true']
test_out_pred = test_history['test']['output_pred']


print("Testing Set -- Classification Report", end="\n\n")
print(classification_report(
    y_true=test_out_true,
    y_pred=test_out_pred,
    target_names=labels
))

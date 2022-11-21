import pandas as pd
import os
import torch
import torch.optim as optim

from utils import (dataset, train, test)
from models.DBN import DBN
from sklearn.metrics import classification_report

DATA_DIR = os.path.join(os.path.abspath('.'), "data")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DBN(
    n_visible=19,
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

# Tran pre-train DBN
print('Start pre-training...')
# model.fit(train_loader)
print('Finished pre-train!')

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

labels = ['Benign', 'Portmap', 'DDos', 'Syn', 'TFTP', 'UDPLag']

# Report
train_output_true = train_history['train']['output_true']
train_output_pred = train_history['train']['output_pred']

print("Training Set -- Classification Report", end="\n\n")
print(classification_report(
    y_true=train_output_true,
    y_pred=train_output_true,
    target_names=labels,
))

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

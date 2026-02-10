import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('.', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    if name not in DATA_HUB:
        raise ValueError(f"{name} does not exist in DATA_HUB.")
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # File is already downloaded and verified
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1048576):
            if chunk:
                f.write(chunk)
    return fname

DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011f4bff10f27d5b857f5db4c5d96')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(numeric_features)
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.astype(float).reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 1)
    )
    return net

def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs=100, learning_rate=0.01, weight_decay=0, batch_size=64):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            net.train()
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        net.eval()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

def plot_loss_curves(train_losses, num_epochs, figsize=(10, 6), 
                     xlabel='epoch', ylabel='log rmse', 
                     title='Training Loss', xlim=[1, 50], yscale='log',
                     legend=['train'], save_path=None):
    """
    Plot loss curves with customizable parameters
    
    Parameters:
    train_losses: list of training losses
    num_epochs: total number of epochs (used for x-axis range)
    figsize: tuple for figure dimensions (width, height)
    xlabel: label for x-axis
    ylabel: label for y-axis
    title: figure title
    xlim: list of [xmin, xmax] for x-axis limits
    yscale: scale for y-axis ('linear', 'log', etc.)
    legend: list of legend labels
    save_path: optional path to save the figure
    """
    # Generate x values (epochs from 1 to num_epochs)
    epochs = np.arange(1, num_epochs + 1)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all loss curves
    for i, loss_curve in enumerate(train_losses):
        ax.plot(epochs, loss_curve, linewidth=2, label=legend[i] if i < len(legend) else f'curve_{i}')
    
    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Set y-axis scale (linear or logarithmic)
    if yscale == 'log':
        ax.set_yscale('log')
    elif yscale == 'linear':
        ax.set_yscale('linear')
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add legend if provided
    if legend:
        ax.legend(fontsize=11)
    
    # Add grid with light transparency
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks as integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    return fig, ax

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                  weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # plot_loss_curves(train_ls, valid_ls, title=f'Fold {i + 1} Training Progress',
        #                     save_path=f'k_fold_{k}_fold_{i + 1}_training_progress.png')
        print(f'fold {i + 1}, train log rmse {train_ls[-1]:f}, '
              f'valid log rmse {valid_ls[-1]:f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse {train_l:f}, '
      f'avg valid log rmse {valid_l:f}')

def train_and_pred(train_features, test_features, train_labels,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                       num_epochs, lr, weight_decay, batch_size)
    # plot_loss_curves([train_ls], num_epochs, xlabel='epoch', ylabel='log rmse',
    #                  title='Training Loss', xlim=[1, num_epochs], yscale='log',
    #                  legend=['train'], save_path='final_training_loss.png')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels,
               num_epochs, lr, weight_decay, batch_size)

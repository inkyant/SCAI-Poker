import copy
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import tqdm
from sklearn.model_selection import train_test_split

import pandas as pd

from query import query 

# get data from csv. Use process_data.py to process pluribus logs first.
csvData = pd.read_csv("Poker_data.csv") 
data_size = 1200

# load full input/output data
X = csvData.iloc[list(range(data_size//2)) + list(range(-data_size//2, 0)), 0:14].to_numpy()
y = csvData.iloc[list(range(data_size//2)) + list(range(-data_size//2, 0)), 14].to_numpy()

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
 

class MLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Linear(14, 8)
        self.layer2 = nn.Linear(8, 6)
        self.layer3 = nn.Linear(6, 1)

    def forward(self, input):
        pred = self.layer1(input)
        pred = f.relu(pred)
        pred = self.layer2(pred)
        pred = f.relu(pred)
        pred = self.layer3(pred)
        pred = f.sigmoid(pred)
        return pred
 

# Define the model
model = MLP()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0003)
 
n_epochs = 150   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
best_epoch = 0
history = []
 
for epoch in range(n_epochs):

    # this just displays the model training as a nice progress bar :)
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")

        # training loop here
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress on the progress bar
            bar.set_postfix(mse=float(loss))
    
    # do not calculate gradients for evaluation of model
    with torch.no_grad():

        # run prediction and store to evaluation history
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)

        # store best evaluation
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
print(f'Best epoch {best_epoch}')
plt.plot(history)
plt.show()


# for fun, run a prediction in the terminal
def pred(cards):
    cards = torch.tensor([cards], dtype=torch.float32)
    return model(cards)
query(pred)
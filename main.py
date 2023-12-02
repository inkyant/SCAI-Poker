import copy
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

import pandas as pd 

# get data
csvData = pd.read_csv("Poker_data.csv") 

data_size = 1500

X = csvData.iloc[list(range(data_size//2)) + list(range(-data_size//2, 0)), 0:14].to_numpy()
y = csvData.iloc[list(range(data_size//2)) + list(range(-data_size//2, 0)), 14].to_numpy()

# print(X.shape)
# print(y.shape)
# print(X)
# print(y)
# print()

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
 
# Define the model
model = nn.Sequential(
    nn.Linear(14, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1),
    nn.Sigmoid()
)
 
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
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
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
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
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

card_dict = {"a": 1, "2": 2, "3": 3, "4": 4, "5": 5,
             "6": 6, "7": 7, "8": 8, "9": 9, "t": 10,
             "j": 11, "q": 12, "k": 13, "ace": 1, "10": 10, "jack": 11, "queen": 12, "king": 13}
suit_dict = {"h": 1, "d": 2, "c": 3, "s": 4, "hearts": 1, "diamonds": 2, "clubs": 3, "spades": 4}

reverse_card_dict = {
    1: "ace",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "jack",
    12: "queen",
    13: "king"
}

reverse_suits_dict = {
    1: "hearts",
    2: "diamonds",
    3: "clubs",
    4: "spades"
}

first = True
while first or input().lower() == 'yes':
    first = False
    cards = []

    while len(cards) < 14:
        
        noCard = True
        while noCard:
            try:
                print('give me a card')
                card = card_dict[input()]
                print('give me a suit')
                suit = suit_dict[input()]
                noCard = False
            except Exception:
                print('Not a card, please try again\n') 

        print('\n')
        cards.append(card)
        cards.append(suit)


    print("your hand cards are: ")
    for i in range(2):
        print(reverse_card_dict[cards[i*2]] + " of " + reverse_suits_dict[cards[i*2 + 1]])

    print("\nyour table cards are: ")
    for i in range(2, 7):
        print(reverse_card_dict[cards[i*2]] + " of " + reverse_suits_dict[cards[i*2 + 1]])

    cards = torch.tensor([cards], dtype=torch.float32)
    pred = model(cards)

    print("\nYour chance of winning is: %.2f" %pred.item())

    print('would you like to continue? [yes]')
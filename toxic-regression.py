import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Load .csv file
np_dataset = np.genfromtxt("qsar_aquatic_toxicity.csv", delimiter=";", skip_header=1)

# Split data by 9:1
train_len = int(round(len(np_dataset) * 0.09) * 10)
test_len = len(np_dataset) - train_len
data_tr = np_dataset[:train_len][:, :-1]
data_ts = np_dataset[train_len:][:, :-1]

# Normalize
mean, std = data_tr.mean(axis=0), data_tr.std(axis=0)
data_tr = (data_tr - mean) / std
data_ts = (data_ts - mean) / std

x_tr, y_tr = data_tr, np_dataset[:train_len][:, -1:]
x_ts, y_ts = data_ts, np_dataset[train_len:][:, -1:]

# Convert Ndarray to Tensor
x_tr, y_tr = torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).float()
x_ts, y_ts = torch.from_numpy(x_ts).float(), torch.from_numpy(y_ts).float()

print("x_tr shape=", x_tr.shape)
print("y_tr shape=", y_tr.shape)
print("x_ts shape=", x_ts.shape)
print("y_ts shape=", y_ts.shape)

# Model
model = torch.nn.Sequential (
    torch.nn.Linear(8, 64), # The number of attributes is 8
    torch.nn.Tanh(),
    torch.nn.Linear(64, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1) # Scalar regression
)

iteration = 4000
learning_rate = 0.00035
weight_decay = 0.6

criterion = torch.nn.MSELoss(reduction='sum') # MSE
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # SGD

# Training loop
train_mse = 0.0
for epoch in range(iteration):
    y_pred = model(x_tr) # Forward pass
    loss = criterion(y_pred, y_tr) # loss 계산
    train_mse = loss.item()
    print(epoch, train_mse)
    if not torch.isfinite(loss): # If loss is infinite
        print("non-finite loss, ending training")
        break
    optimizer.zero_grad() # Change degree into 0
    loss.backward() # Backward pass, calculate the gredient of loss for all learnable parameters of the model
    optimizer.step() # Update weights

# Testing

""" # Model test for y_tr
y_pred = model(x_tr)
loss = criterion(y_pred, y_tr)
test_mse = loss.item()
print("@ test mse =", test_mse)
plt.scatter([x[0] for x in y_tr.detach().numpy()], [y[0] for y in y_pred.detach().numpy()], s=10)
plt.xlabel("True y_tr")
plt.ylabel("Predicted y_tr")
plt.title("Iteration=" + str(iteration) + " | Learning rate=" + str(learning_rate) + " | Weight decay=" + str(weight_decay))
plt.text(0.5, 9.5, "Train MSE=" + str(train_mse) + " | Test MSE=" + str(test_mse))
"""
# Model test for y_ts
y_pred = model(x_ts)
loss = criterion(y_pred, y_ts)
test_mse = loss.item()
print("@ test mse =", test_mse)
plt.scatter([x[0] for x in y_ts.detach().numpy()], [y[0] for y in y_pred.detach().numpy()], s=10)
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.title("Iteration=" + str(iteration) + " | Learning rate=" + str(learning_rate) + " | Weight decay=" + str(weight_decay))
plt.text(0.5, 9.5, "Train MSE=" + str(train_mse) + " | Test MSE=" + str(test_mse))

plt.axline((-100, -100), (100, 100), c="grey", lw=1)
plt.ylim(0, 10)
plt.xlim(0, 10)
plt.show(block=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from tqdm import tqdm
import os

psi_star = np.array([0, 0, 0])

"""
a = y - x
b = y + x
c = z - y
d = z + y
e = x - z
f = x + z
g = 2y - 3x
h = 2y + 3x
i = 2z - 3y
j = 2z + 3y
k = 2x - 3z
l = 2x + 3z

Implies y = (a + b)/2, (d - c)/2, (g + h)/4, (j - i)/6, average these for true y
Similarly for x, z
"""

### STUDY SWITCHES ###
USE_SCHEDULE = False
SIMPLE = True
SIX = False
TWELVE = False

if SIMPLE:
    D = 3
elif SIX:
    D = 6
elif TWELVE:
    D = 12

### SIMULATOR ###
def dim_redux(vec, D):
    if D == 3:
        return vec
    elif D == 6:
        a, b, c, d, e, f = vec
        x = ((b-a)/2 + (e+f)/2)/2
        y = ((a+b)/2 + (d-c)/2)/2
        z = ((c+d)/2 + (f-e)/2)/2
    elif D == 12:
        a, b, c, d, e, f, g, h, i, j, k, l = vec
        x = ((b-a)/2 + (e+f)/2 + (h-g)/6 + (l+k)/4)/4
        y = ((a+b)/2 + (d-c)/2 + (g+h)/4 + (j-i)/6)/4
        z = ((c+d)/2 + (f-e)/2 + (i+j)/4 + (l-k)/6)/4
    return x, y, z

def true_function(vec, D=3):
    params = np.array(dim_redux(vec, D))
    return np.sum(np.square(params - psi_star))

def simulator(vec, noise_std=0):
    noise = np.random.normal(0, noise_std)
    return true_function(vec, D=D) + noise

### SURROGATE MODEL ###
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()

        if SIMPLE:
            self.model = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif SIX:
            self.model = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif TWELVE:
            self.model = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
    def forward(self, x):
        return self.model(x)

### HYPER-HYPERPARAMETERS ###
N = 100
M = 10
K = 1
eps = .3 / np.sqrt(D)

max_iter = 200
batch_size = int(N*1)
num_epochs = 1
adam_learning_rate = .0432
psi_learning_rate = .5*adam_learning_rate
max_history_len = int(N*max_iter*1)
convergence_radius = 0
# Step 1
if SIMPLE:
    psi = np.array([-.5, .5, 1])
elif SIX:
    psi = np.array([0, 2, 0, 2, 0, 2], dtype="float64")  

### INITIALIZE HISTORY, MODEL, & OPTIMIZER ###
history = deque(maxlen=max_history_len)
psi_history = [psi.copy()]

surrogate_model = SurrogateModel()
optimizer = optim.Adam(surrogate_model.parameters(), lr=adam_learning_rate, betas=(0.9, 0.999), eps=1e-08)
if USE_SCHEDULE:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
loss_func = nn.MSELoss()

# Step 2
iters = 0
not_converged = True
with tqdm(total=max_iter, desc="Total Iterations") as pbar_outer:  
    while not_converged:

        history = deque(maxlen=max_history_len)
        surrogate_model = SurrogateModel()
        optimizer = optim.Adam(surrogate_model.parameters(), lr=adam_learning_rate, betas=(0.9, 0.999), eps=1e-08)
        if USE_SCHEDULE:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        loss_func = nn.MSELoss()

        psi_samples = psi + np.random.uniform(-eps, eps, size=(N, D))  # Step 3
        psi_repeated = np.repeat(psi_samples, M, axis=0)  # Repeat each psi_i M times for Step 4
        sim_samples = np.array([simulator(vec) for vec in psi_repeated])  # Step 5
        history.extend(zip(psi_samples, sim_samples))  # Step 6

        # Step 7
        X_train_np = np.array([sample[0] for sample in history])  # Creating a tensor from a list
        y_train_np = np.array([sample[1] for sample in history])  # of np.arrays is extremely slow
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Step 8
        surrogate_model.train()
        with tqdm(total=num_epochs, desc="Epochs", leave=False) as pbar_inner:
            for epoch in range(num_epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    predictions = surrogate_model(X_batch)
                    loss = loss_func(predictions, y_batch)
                    loss.backward()
                    optimizer.step()
                pbar_inner.update(1)
            if USE_SCHEDULE:
                scheduler.step()
        surrogate_model.eval()  # Step 9

        # Step 10
        psi_tensor = torch.tensor(psi, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        y_pred_samples = surrogate_model(psi_tensor)

        # Step 11
        grad_outputs = torch.autograd.grad(outputs=y_pred_samples, inputs=psi_tensor,
                                           grad_outputs=torch.ones_like(y_pred_samples), create_graph=True)[0]
        grad_psi = grad_outputs.detach().numpy().flatten()

        # Step 12
        psi -= psi_learning_rate * grad_psi
        psi_history.append(psi.copy())
        
        # Step 13 
        infinity_norm = np.linalg.norm(psi_learning_rate * grad_psi, ord=np.inf)
        if infinity_norm < convergence_radius:
            break
        iters += 1
        pbar_outer.update(1)
        if iters > max_iter:
            break

# Plotting function
def plot_history(psi_history):
    psi_history = np.array(psi_history)
    loss = true_function(psi_history[-1], D=D)
    print(loss)
    for i in range(D):
        if SIMPLE:
            plt.plot(psi_history[:, i], label=["x", "y", "z"][i])
        else:
            plt.plot(psi_history[:, i], label=["y - x", "y + x", "z - y", "z + y", 
                                               "x - z", "x + z", "2y - 3x",  "2y + 3x", 
                                               "2z - 3y", "2z + 3y", "2x - 3z", "2x + 3z"][i])
    plt.xlabel('Iteration')
    plt.ylabel('Parameters')
    plt.legend()
    plt.title(f'D {D} Adam LR {adam_learning_rate:.4f}, Param LR {psi_learning_rate:.4f}, Loss {loss:.8f}, M {M}')
    plt.grid(True)

    ymin, ymax = plt.ylim()
    plt.ylim(min(ymin, np.min(psi_star)-0.1), max(ymax, np.max(psi_star)+.1))

    plot_path = f"clean_plots/{D}/{adam_learning_rate:.2f}/{psi_learning_rate:.2f}"
    os.makedirs(plot_path, exist_ok=True)

    plt.savefig(f"{plot_path}/{loss}.png")

# Plot psi history
plot_history(psi_history)
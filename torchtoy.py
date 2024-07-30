import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# Define the true function and the noisy simulator
def true_function(x, y, z):
    return x**2 + y**2 + z**2

def simulator(x, y, z, noise_std=0):
    noise = np.random.normal(0, noise_std)
    return true_function(x, y, z) + noise

# MLP Surrogate model
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize parameters
psi = np.array([1.0, 1.0, 1.0])
epsilon = 0.1
N = 10
iterations = 1000
learning_rate = .1
max_history_size = 500
batch_size = 32

# History for storing samples and psi values
history = deque(maxlen=max_history_size)
psi_history = []

# Create the surrogate model and optimizer
surrogate_model = SurrogateModel()
optimizer = optim.Adam(surrogate_model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

for _ in tqdm(range(iterations), desc='Optimizing'):
    # Store the current psi
    psi_history.append(psi.copy())
    
    # Sample new psi values in the local neighborhood
    psi_samples = psi + np.random.uniform(-epsilon, epsilon, size=(N, 3))

    # Evaluate the simulator for each sampled psi
    samples = np.array([simulator(x, y, z) for x, y, z in psi_samples])
    history.extend(zip(psi_samples, samples))

    # Prepare training data for the surrogate model
    X_train_np = np.array([sample[0] for sample in history])
    y_train_np = np.array([sample[1] for sample in history])
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    
    # Train using mini-batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        surrogate_model.train()
        optimizer.zero_grad()
        predictions = surrogate_model(X_batch)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        optimizer.step()

    # Compute the gradient of the surrogate model at the current psi
    surrogate_model.eval()
    psi_tensor = torch.tensor(psi, dtype=torch.float32, requires_grad=True).view(1, -1)
    prediction = surrogate_model(psi_tensor)
    prediction.backward()
    gradient = psi_tensor.grad

    if gradient is not None:
        gradient = gradient.numpy().flatten()
        print("gay")
        raise AssertionError
    else:
        # Use retain_grad() to retain the gradient for non-leaf tensors
        psi_tensor.retain_grad()
        prediction = surrogate_model(psi_tensor)
        prediction.backward()
        gradient = psi_tensor.grad.numpy().flatten()

    # Update parameters using Adam optimizer
    psi -= learning_rate * gradient

# Convert psi_history to a numpy array for easier manipulation
psi_history = np.array(psi_history)

# Plot the learning process
def plot_learning(psi_history):
    plt.figure(figsize=(12, 6))
    plt.plot(psi_history[:, 0], label='x')
    plt.plot(psi_history[:, 1], label='y')
    plt.plot(psi_history[:, 2], label='z')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Learning Process of Parameters')
    plt.legend()
    plt.grid(True)
    plt.savefig("torch_learning.png")

# Call the function to plot the learning process
plot_learning(psi_history)
print(psi_history[-1])

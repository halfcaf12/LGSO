import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Define the true function and the noisy simulator
def true_function(x, y, z):
    return x**2 + y**2 + z**2

def simulator(x, y, z, noise_std=0.1):
    noise = np.random.normal(0, noise_std)
    return true_function(x, y, z) + noise

# Initialize parameters
psi = np.array([1.0, 1.0, 1.0])
epsilon = 0.2
N = 10
max_iters = 300
min_update = .001
learning_rate = 0.025

# History for storing samples and psi values
history = []
psi_history = []

iters = 0
update = np.sum(psi)/3
while update > min_update:
    if iters > max_iters:
        break

    # Store the current psi
    psi_history.append(psi.copy())
    
    # Sample new psi values in the local neighborhood
    psi_samples = psi + np.random.uniform(-epsilon, epsilon, size=(N, 3))

    # Evaluate the simulator for each sampled psi
    samples = np.array([simulator(x, y, z) for x, y, z in psi_samples])
    history.extend(zip(psi_samples, samples))

    # Prepare training data for the surrogate model
    X_train = np.array([sample[0] for sample in history])
    y_train = np.array([sample[1] for sample in history])

    # Train a Gaussian Process surrogate model
    kernel = RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X_train, y_train)

    # Compute finite difference gradient
    gradient = np.zeros_like(psi)
    delta = 1e-5
    for i in range(len(psi)):
        psi_plus = psi.copy()
        psi_minus = psi.copy()
        psi_plus[i] += delta
        psi_minus[i] -= delta
        y_pred_plus = gp.predict([psi_plus])[0]
        y_pred_minus = gp.predict([psi_minus])[0]
        gradient[i] = (y_pred_plus - y_pred_minus) / (2 * delta)
    
    # Update parameters using SGD
    psi -= learning_rate * gradient
    update = np.sum(learning_rate * gradient)/3

    iters += 1

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
    plt.savefig("learning.png")

# Call the function to plot the learning process
plot_learning(psi_history)
print(psi_history[-1])
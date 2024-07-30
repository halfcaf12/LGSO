#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import uproot
import subprocess
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from tqdm import tqdm
import os

srcDir = Path(__file__).resolve().parent

def run_ckf(params, names, outDir):
    if len(params) != len(names):
        raise Exception("Length of Params must equal names")

    ckf_script = srcDir / "ckf.py"
    nevts = "--nEvents=1"
    indir = "--indir=" + str(srcDir)
    outdir = "--output=" + str(outDir)

    ret = ["python"]
    ret.append(ckf_script)
    ret.append(nevts)
    ret.append(indir)
    ret.append(outdir)

    i = 0
    for param in params:
        if i == 0:
            param = int(param)
        arg = "--sf_" + names[i] + "=" + str(param)
        ret.append(arg)
        i += 1

    # Run CKF for the given parameters
    subprocess.call(ret)


class Objective:
    def __init__(self, k_dup, k_time):
        self.res = {
            "eff": [],
            "fakerate": [],
            "duplicaterate": [],
            "runtime": [],
        }

        self.k_dup = k_dup
        self.k_time = k_time

    def __call__(self, params, ckf_perf=True):
        keys = [
            "maxSeedsPerSpM",
            "cotThetaMax",
            "sigmaScattering",
            "radLengthPerSeed",
            "impactMax",
            "maxPtScattering",
            "deltaRMin",
            "deltaRMax",
        ]
        # params[0] = int(params[0])
        get_tracking_perf(self, ckf_perf, params, keys)

        efficiency = self.res["eff"][-1]
        penalty = (
            self.res["fakerate"][-1]
            + self.res["duplicaterate"][-1] / self.k_dup
            + self.res["runtime"][-1] / self.k_time
        )

        return efficiency - penalty


def get_tracking_perf(self, ckf_perf, params, keys):
    if ckf_perf:
        outDirName = "Output_CKF"
        outputfile = srcDir / outDirName / "performance_ckf.root"
        effContName = "particles"
        contName = "tracks"
    else:
        outDirName = "Output_Seeding"
        outputfile = srcDir / outDirName / "performance_seeding.root"
        effContName = "seeds"
        contName = "seeds"

    outputDir = Path(srcDir / outDirName)
    outputDir.mkdir(exist_ok=True)
    run_ckf(params, keys, outputDir)
    rootFile = uproot.open(outputfile)
    self.res["eff"].append(rootFile["eff_" + effContName].member("fElements")[0])
    self.res["fakerate"].append(rootFile["fakerate_" + contName].member("fElements")[0])
    self.res["duplicaterate"].append(
        rootFile["duplicaterate_" + contName].member("fElements")[0]
    )

    timingfile = srcDir / outDirName / "timing.tsv"
    timing = pd.read_csv(timingfile, sep="\t")

    if ckf_perf:
        time_ckf = float(
            timing[timing["identifier"].str.match("Algorithm:TrackFindingAlgorithm")]["time_perevent_s"].iloc[0]
        )

    time_seeding = float(
        timing[timing["identifier"].str.match("Algorithm:SeedingAlgorithm")]["time_perevent_s"].iloc[0]
    )

    if ckf_perf:
        self.res["runtime"].append(time_ckf + time_seeding)
    else:
        self.res["runtime"].append(time_seeding)

# Plotting function


def main():
    def plot_history(psi_history):
        print(f"maxSeedsPerSpM: {psi[0]}") 
        print(f"cotThetaMax: {psi[1]}") 
        print(f"sigmaScattering: {psi[2]}")
        print(f"radLengthPerSeed: {psi[3]}")
        print(f"impactMax: {psi[4]}")
        print(f"maxPtScattering: {psi[5]}")
        print(f"deltaRMin: {psi[6]}")
        print(f"deltaRMax: {psi[7]}")
        loss = objective(psi)
        print(loss)

        psi_history = np.array(psi_history)
        for i in range(D):
            plt.plot(psi_history[:, i], label=["maxSeedsPerSpM", "cotThetaMax", "sigmaScattering", 
                                               "radLengthPerSeed", "impactMax", "maxPtScattering",
                                               "deltaRMin", "deltaRMax"][i])
        plt.xlabel('Iteration')
        plt.ylabel('Parameters')
        plt.legend()
        plt.title(f'D {D} Adam LR {adam_learning_rate}, Param LR {psi_learning_rate}, Loss {loss}')
        plt.grid(True)

        plot_path = f"clean_plots/{D}/{adam_learning_rate:.2f}/{psi_learning_rate:.2f}"
        os.makedirs(plot_path, exist_ok=True)

        plt.savefig(f"{plot_path}/{loss}.png")

    k_dup = 5
    k_time = 5

    # Initializing the objective (score) function
    objective = Objective(k_dup, k_time)

    """
    start_values = {
        "maxSeedsPerSpM": 1,
        "cotThetaMax": 7.40627,
        "sigmaScattering": 50,
        "radLengthPerSeed": 0.1,
        "impactMax": 3.0,
        "maxPtScattering": 10.0,
        "deltaRMin": 1.0,
        "deltaRMax": 60.0
}

    """
    D = 8
    ### SURROGATE MODEL ###
    class SurrogateModel(nn.Module):
        def __init__(self):
            super(SurrogateModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(D, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
        def forward(self, x):
            return self.model(x)

    ### HYPER-HYPERPARAMETERS ###
    N = 80
    M = 5
    K = 1
    eps = 3 / np.sqrt(D)

    max_iter = 5
    batch_size = int(N*1)
    num_epochs = 1
    adam_learning_rate = .0432
    psi_learning_rate = 5
    max_history_len = int(N*max_iter*1)
    convergence_radius = 1e-8
    # Step 1
    psi = np.array([1, 7.40627, 50, 0.1, 3.0, 10.0, 1.0, 60.0])

    ### INITIALIZE HISTORY, MODEL, & OPTIMIZER ###
    psi_history = [psi.copy()]
    
    history = deque(maxlen=max_history_len)
    surrogate_model = SurrogateModel()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=adam_learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # if USE_SCHEDULE:
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_func = nn.MSELoss()

    # Step 2
    iters = 0
    not_converged = True
    with tqdm(total=max_iter, desc="Total Iterations") as pbar_outer:  
        while not_converged:  
            # history = deque(maxlen=max_history_len)
            # surrogate_model = SurrogateModel()
            # optimizer = optim.Adam(surrogate_model.parameters(), lr=adam_learning_rate, betas=(0.9, 0.999), eps=1e-08)
            # loss_func = nn.MSELoss()

            psi_samples = psi + np.random.uniform(-eps, eps, size=(N, D))  # Step 3
            psi_repeated = np.repeat(psi_samples, M, axis=0)  # Repeat each psi_i M times for Step 4
            sim_samples = np.array([objective(params) for params in psi_repeated])  # Step 5

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
                # if USE_SCHEDULE:
                #     scheduler.step()
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
                plot_history(psi_history)
                break
            iters += 1
            pbar_outer.update(1)
            if iters > max_iter:
                plot_history(psi_history)
                break

if __name__ == "__main__":
    main()

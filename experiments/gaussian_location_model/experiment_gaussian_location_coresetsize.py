import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mcmc_sampler.models.gaussian_location as gaussian_location

"""
This experiment illustrates the effect of the coreset size on the quality of the coreset in the Gaussian location 
model. The algorithm is tested with the following coreset sizes M dependent on the size N of the full dataset:
M in {floor(N/1000), floor(N/500), floor(N/100), floor(N/50), floor(N/10)}
One run is started for each coreset size. The Kullback-Leibler divergence is plotted for the corresponding coreset size.
Furthermore, the computational cost (M+S)*K*t is also illustrated for each choice of M.
This gives us two plots, one where the x-axis is the coreset size and the y-axis is the KL divergence. 
The second, where the x-axis is the cost, the y-axis is the KL and a graph for each coreset size.   
"""

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"experiment_coreset_size_Gaussian_log_{timestamp}.txt"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info(f"Start: Experiment Gaussian Location Model - Coreset Size")
    print("Running...")
    T = 25000
    d = 15
    N = 50000
    coreset_sizes = [int(np.floor(N / 1000)),
                     int(np.floor(N / 500)),
                     int(np.floor(N / 100)),
                     int(np.floor(N / 50)),
                     int(np.floor(N / 10))]
    S = N
    K = 2
    np.random.seed(42)
    subsample = True
    replace = False

    # Generate synthetic data set
    data = np.random.randn(d, N)

    kl_divergence = []

    for i, M in enumerate(coreset_sizes):
        coreset = data[:, :M]
        kls = gaussian_location.run_coresetmcmc(data=data,
                                                S=S,
                                                K=K,
                                                d=d,
                                                coreset=coreset,
                                                T=T,
                                                subsample=subsample,
                                                replace=replace)

        cost_values = (M + S) * K * np.arange(1, T + 1)

        num_points = len(cost_values)
        indices = np.unique(np.logspace(0,
                                        np.log10(num_points - 1),
                                        num=50,
                                        dtype=int))

        sampled_costs = np.array(cost_values)[indices]
        sampled_kls = np.array(kls)[indices]

        # First plot (KL Divergence vs. Cost)
        # Proportion of the corset of the full data set
        proportion = [1000, 500, 100, 50, 10]
        sns.lineplot(x=sampled_costs,
                     y=sampled_kls,
                     marker="x",
                     markersize=5,
                     ax=plt.gca(),
                     label=f"Coreset Size = N/{proportion[i]}",
                     color=sns.color_palette("inferno")[i])

        kl_divergence.append(kls[-1])
        logging.info(f"Coreset size: {M}, "
                     f"results: {kls[-1]}")

    save_folder = "../../plots/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.xlabel("Cost")
    plt.ylabel("KL-Divergence")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(save_folder,
                f"experiment_coreset_size_cost_Gaussian-location_{timestamp}.svg"),
                format="svg")

    # Second plot (KL Divergence vs. Coreset Size)
    proportion = [1000, 500, 100, 50, 10]
    x_labels = [f"N/{size}" for size in proportion]

    sns.lineplot(x=x_labels,
                 y=kl_divergence,
                 marker="o",
                 markersize=5,
                 color=sns.color_palette("binary")[-1])
    plt.xlabel("Coreset Size")
    plt.ylabel("KL Divergence")
    plt.yscale("log")
    plt.savefig(os.path.join(save_folder,
                f"experiment_coreset_size_kl_Gaussian-location_{timestamp}.svg"),
                format="svg")

    print("Experiment completed.")

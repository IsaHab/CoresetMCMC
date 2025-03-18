import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import mcmc_sampler.models.gaussian_location as gaussian_location

"""
This experiment compares a simple weighted subset construction algorithm (UniformSubsampling) with the CoresetMCMC.
Therefore the Gaussian location model is tested with the following coreset sizes M dependent on 
the size N of the full dataset:
M in {floor(N/1000), floor(N/500), floor(N/100), floor(N/50), floor(N/10)}
This is done for the CoresetMCMC and the UniformSubsampling. One run is started for each algorithm and for each coreset 
size. The Kullback-Leibler divergence is plotted for the corresponding algorithm and coreset size. This gives us one 
plot where the x-axis is the coreset size, the y-axis is the KL divergence. The plot contains two graphs, one for 
CoresetMCMC and one for UniformSubsampling.
"""

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"experiment_unif_Gaussian_log_{timestamp}.txt"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info(f"Start: Experiment Gaussian Location Model - Unifor Sampling")
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

    kl_divergence_unif = []
    kl_divergence = []

    for i, M in enumerate(tqdm(coreset_sizes, desc="Step")):
        coreset = data[:, :M]
        kls_unif = gaussian_location.run_unif(data=data,
                                              d=d,
                                              coreset=coreset,
                                              T=T)
        kls = gaussian_location.run_coresetmcmc(data=data,
                                                S=S,
                                                K=K,
                                                d=d,
                                                coreset=coreset,
                                                T=T,
                                                subsample=subsample,
                                                replace=replace)
        kl_divergence_unif.append(kls_unif[-1])
        kl_divergence.append(kls[-1])
        logging.info(f"Unif Subsampling: Coresetsize: {M},"
                     f"results: {kls_unif[-1]}")
        logging.info(f"CoresetMCMC: Coresetsize: {M},"
                     f"results: {kls[-1]}")

    save_folder = "../../plots/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    proportion = [1000, 500, 100, 50, 10] # Proportion of the corset of the full data set
    x_labels = [f"N/{size}" for size in proportion]

    sns.lineplot(x=x_labels,
                 y=kl_divergence,
                 label=f"CoresetMCMC",
                 marker="o",
                 markersize=5,
                 color=sns.color_palette("binary")[-1])
    sns.lineplot(x=x_labels,
                 y=kl_divergence_unif,
                 label=f"Uniform Subsampling",
                 marker="o",
                 markersize=5,
                 color=sns.color_palette("Wistia")[-1])
    plt.title("Gaussian Location Model")
    plt.xlabel("Coreset Size")
    plt.ylabel("KL Divergence")
    plt.yscale("log")
    plt.savefig(os.path.join(save_folder,
                f"experiment_unif_Gaussian-location_{timestamp}.svg"),
                format="svg")

    print("Experiment completed.")

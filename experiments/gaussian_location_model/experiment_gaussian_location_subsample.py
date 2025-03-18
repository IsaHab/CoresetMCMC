import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import mcmc_sampler.models.gaussian_location as gaussian_location

"""
This experiment illustrates the effect of the subsample size on the quality of the coreset in the Gaussian
location model. The CoresetMCMC algorithm is tested with the following subsample sizes S where M is 
the number of coreset data points and N is the size of the full dataset:
S in {1, M, 5M, 10, N} and M in {floor(N/1000), floor(N/500), floor(N/100), floor(N/50), floor(N/10)}.
One run is started for each coreset size and each subsample size. The Kullback-Leibler divergence is plotted for the 
corresponding coreset size and choice of S. This gives us one plot where the x-axis is the coreset size, the y-axis is 
the KL divergence and a graph for each choice of S.  
"""


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"experiment_subsample_size_Gaussian_log_{timestamp}.txt"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info(f"Start: Experiment Gaussian Location Model - Subsample Size")
    print("Running...")
    T = 25000
    d = 15
    N = 50000
    coreset_sizes = [int(np.floor(N/1000)),
                     int(np.floor(N/500)),
                     int(np.floor(N/100)),
                     int(np.floor(N/50)),
                     int(np.floor(N/10))]
    K = 2
    np.random.seed(42)
    subsample = True
    replace = False

    # Generate synthetic data set
    data = np.random.randn(d, N)
    df_list = []

    subsample_size = [0, 1, 5, 10, N] # [1, 2M, 5M, 10M, N]
    proportion = [1000, 500, 100, 50, 10]
    x_labels = [f"N/{size}" for size in proportion]

    for i, s in enumerate(subsample_size):
        kl_divergence = []
        for M in tqdm(coreset_sizes):
            subs = s * M
            if s == 0:
                subs = 1
            if s == N:
                subs = N

            coreset = data[:, :M]
            kls = gaussian_location.run_coresetmcmc(data=data,
                                                    S=subs,
                                                    K=K,
                                                    d=d,
                                                    coreset=coreset,
                                                    T=T,
                                                    subsample=subsample,
                                                    replace=replace)
            kl_divergence.append(kls[-1])

            logging.info(f"Coresetsize: {M}, "
                         f"subsample size: {s}, "
                         f"results: {kls[-1]}")

        s_label = [f"{s}M"]
        color = sns.color_palette("flare", len(subsample_size))[i]
        if s == 0:
            s_label = [1]
        if s == N:
            s_label = [f"N"]
            color = sns.color_palette("binary")[-1]

        dat = {
            "Coreset Size": x_labels,
            "KL Divergence": kl_divergence,
            "Subsample Size": s_label * len(proportion),
            "Color": [color] * len(proportion)
        }
        df_list.append(pd.DataFrame(dat))

    df = pd.concat(df_list, ignore_index=True)

    save_folder = "../../plots/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    marker = ["D", "X", "s", "P", "o"]
    line_style = ["-", "--", ":", "-.", "-"]
    i = 0
    for s_label, group in df.groupby("Subsample Size"):
        sns.lineplot(
            data=group,
            x="Coreset Size",
            y="KL Divergence",
            label=f"{s_label}",
            marker= marker[i],
            linestyle=line_style[i],
            color=group["Color"].iloc[0]
        )
        i = i + 1
    plt.xlabel("Coreset Size")
    plt.ylabel("KL Divergence")
    plt.legend(title="Subsample Size")
    plt.yscale("log")
    plt.savefig(os.path.join(save_folder,
                f"experiment_subsample_size_Gaussian-location_{timestamp}.svg"),
                format="svg")

    print("Experiment completed.")

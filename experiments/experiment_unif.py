import logging
import os
from datetime import datetime

import experiment_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mcmc_sampler.kernels import HMC, MALA, SliceSampler

"""
This experiment compares a simple weighted subset construction algorithm (UniformSubsampling) with the CoresetMCMC.
Therefore each regression model (linear, logistic, poisson) is tested with the following coreset sizes M dependent on 
the size N of the full dataset:
M in {floor(N/1000), floor(N/500), floor(N/100), floor(N/50), floor(N/10)}
This is done for the CoresetMCMC and the UniformSubsampling.
Five independent runs (Z=5) are started for each algorithm for each regression model for each coreset size. 
The mean of the two moment Kullback-Leibler divergence of these five runs is plotted for the corresponding model, 
coreset size. This gives us three plots (one for each regression model) where the x-axis is the coreset size, the y-axis
is the two moment KL divergence. Each plot contains two graphs, one for CoresetMCMC and one for UniformSubsampling.
"""

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"experiment_unif_log_{timestamp}.txt"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info(f"Start: Uniform Subsampling Experiment")
    save_folder = "../plots/"

    proportion = [1000, 500, 100, 50, 10] # Proportion of the corset of the full data set
    kernel = {"Slice Sampler": SliceSampler(),
              "HMC": HMC(),
              "MALA": MALA()}

    models = {"Linear Regression": "linear_regression",
              "Logistic Regression": "logistic_regression",
              "Poisson Regression": "poisson_regression"}
    Z = 5

    for kernel_name, kernel in kernel.items():
        for model_name, model in models.items():
            results_unif = []
            results = []
            for z in range(0, Z):
                kl_est_coreset_size_unif = experiment_utils.run_experiment_unif(model_type=model,
                                                                                mcmc_kernel=kernel,
                                                                                frac=proportion)
                kl_est_coreset_size = experiment_utils.run_experiment_csize(model_type=model,
                                                                            mcmc_kernel=kernel,
                                                                            frac=proportion)

                if not results_unif:
                    results_unif = kl_est_coreset_size_unif
                else:
                    results_unif = [x + y for x, y in zip(results_unif, kl_est_coreset_size_unif)]

                if not results:
                    results = kl_est_coreset_size
                else:
                    results = [x + y for x, y in zip(results, kl_est_coreset_size)]

                logging.info(f"Uniform Subs: Kernel: {kernel_name}, "
                             f"Model: {model}, "
                             f"Z: {z}, "
                             f"results: {results_unif}")
                logging.info(f"CoresetMCMC: Kernel: {kernel_name}, "
                             f"Model: {model}, "
                             f"Z: {z}, "
                             f"results: {results}")

            results_unif = [x / Z for x in results_unif]
            results = [x / Z for x in results]

            x_labels = [f"N/{size}" for size in proportion]

            dat = {
                "Coreset Size": x_labels,
                "KL Divergence": results_unif,
                "Model": [f"{model_name}"] * len(proportion)
            }
            df_unif = pd.DataFrame(dat)

            dat["Coreset Size"] = x_labels
            dat["KL Divergence"] = results
            dat["Model"] = [f"{model_name}"] * len(proportion)
            df_ = pd.DataFrame(dat)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if model_name == "Linear Regression":
                color_ = "b"
            elif model_name == "Logistic Regression":
                color_ = "r"
            else:
                color_ = "g"

            plt.figure()
            sns.lineplot(data=df_unif,
                         x="Coreset Size",
                         y="KL Divergence",
                         label=f"Uniform Subsampling",
                         markers=True,
                         color=sns.color_palette("Wistia")[-1])
            sns.lineplot(data=df_,
                         x="Coreset Size",
                         y="KL Divergence",
                         label=f"CoresetMCMC",
                         markers=True,
                         color=color_)
            plt.title(model_name)
            plt.xlabel("Coreset Size")
            plt.ylabel("Two-Moment KL Divergence")
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder,
                        f"experiment_unif_{kernel_name}_{model_name}_{timestamp}.svg"),
                        format="svg")

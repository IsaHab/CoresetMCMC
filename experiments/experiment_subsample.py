import logging
import os
from datetime import datetime

import experiment_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mcmc_sampler.kernels import HMC, MALA, SliceSampler

"""
This experiment illustrates the effect of the subsample size on the quality of the coreset.
Therefore each regression model (linear, logistic, poisson) is tested with the following subsample sizes S where M is 
the number of coreset data points and N is the size of the full dataset:
S in {1, M, 5M, 10, N} and M in {floor(N/1000), floor(N/500), floor(N/100), floor(N/50), floor(N/10)}.
Five independent runs (Z=5) are started for each regression model for each coreset size for each subsample size. 
The mean of the two moment Kullback-Leibler divergence of these five runs is plotted for the corresponding model,
coreset size, and choice of S. This gives us three plots (one for each regression model) where the x-axis is the 
coreset size, the y-axis is the two moment KL divergence and a graph for each choice of S. 
"""

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"experiment_subsample_size_log_{timestamp}.txt"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info(f"Start: Subsample Experiment {timestamp}")

    # Proportion of the corset of the full data set
    proportion = [1000, 500, 100, 50, 10]
    # Factor x for subsample size S=x*M
    subsample_size = [1, 5, 10]
    kernel = {"Slice Sampler": SliceSampler(),
              "HMC": HMC(),
              "MALA": MALA()}
    models = {"Linear Regression": "linear_regression",
              "Logistic Regression": "logistic_regression",
              "Poisson Regression": "poisson_regression"}

    Z = 5

    for kernel_name, kernel in kernel.items():
        for model_name, model in models.items():
            results = {"size = 1": [],
                       "size = 1M": [],
                       "size = 5M": [],
                       "size = 10M": [],
                       "size = N": []}
            for z in range(0, Z):
                kl_est_coreset_size = experiment_utils.run_experiment_csize(model_type=model,
                                                                            mcmc_kernel=kernel,
                                                                            frac=proportion,
                                                                            subsampling=[[True, 1]])
                if not results["size = 1"]:
                    results["size = 1"] = kl_est_coreset_size
                else:
                    results["size = 1"] = [x + y for x, y in zip(results["size = 1"], kl_est_coreset_size)]
                logging.info(f"Kernel: {kernel_name}, "
                             f"Model: {model_name}, "
                             f"Subs_Size: {"size = 1"}, "
                             f"Z: {z}, "
                             f"results: {results["size = 1"]}")

                kl_est_coreset_size = experiment_utils.run_experiment_csize(model_type=model,
                                                                            mcmc_kernel=kernel,
                                                                            frac=proportion)
                if not results["size = N"]:
                    results["size = N"] = kl_est_coreset_size
                else:
                    results["size = N"] = [x + y for x, y in zip(results["size = N"], kl_est_coreset_size)]

                logging.info(f"Kernel: {kernel_name}, "
                             f"Model: {model_name}, "
                             f"Subs_Size: {"size = N"}, "
                             f"Z: {z}, "
                             f"results: {results["size = N"]}")

                for subs in subsample_size:
                    kl_est_coreset_size = experiment_utils.run_experiment_subssize(model_type=model,
                                                                                   mcmc_kernel=kernel,
                                                                                   frac=proportion,
                                                                                   subsampling=subs)
                    key = f"size = {subs}M"
                    if not results[key]:
                        results[key] = kl_est_coreset_size
                    else:
                        results[key] = [x + y for x, y in zip(results[key], kl_est_coreset_size)]

                    logging.info(f"Kernel: {kernel_name}, "
                                 f"Model: {model_name}, "
                                 f"Subs_Size:: {key}, "
                                 f"Z: {z}, "
                                 f"results: {results[key]}")
            results = {size: [x / Z for x in values] for size, values in results.items()}

            logging.info(f"Kernel: {kernel_name}, "
                         f"Model: {model_name}, "
                         f"final result: {results}")

            x_labels = [f"N/{size}" for size in proportion]

            dat = {
                "Coreset Size": x_labels,
                "KL Divergence": results["size = 1"],
                "Kernel": ["Slice Sampler"] * len(proportion),
                "Subsample Size": ["1"] * len(proportion),
                "Model": [f"{model_name}"] * len(proportion)
            }
            df_one = pd.DataFrame(dat)

            dat["KL Divergence"] = results["size = 1M"]
            dat["Kernel"] = [f"{kernel_name}"] * len(proportion)
            dat["Subsample Size"] = ["M"] * len(proportion)
            dat["Model"] = [f"{model_name}"] * len(proportion)
            df_M = pd.DataFrame(dat)

            dat["KL Divergence"] = results["size = 5M"]
            dat["Kernel"] = [f"{kernel_name}"] * len(proportion)
            dat["Subsample Size"] = ["5M"] * len(proportion)
            dat["Model"] = [f"{model_name}"] * len(proportion)
            df_fiveM = pd.DataFrame(dat)

            dat["KL Divergence"] = results["size = 10M"]
            dat["Kernel"] = [f"{kernel_name}"] * len(proportion)
            dat["Subsample Size"] = ["10M"] * len(proportion)
            dat["Model"] = [f"{model_name}"] * len(proportion)
            df_tenM = pd.DataFrame(dat)

            dat["KL Divergence"] = results["size = N"]
            dat["Kernel"] = [f"{kernel_name}"] * len(proportion)
            dat["Subsample Size"] = ["N"] * len(proportion)
            dat["Model"] = [f"{model_name}"] * len(proportion)
            df_N = pd.DataFrame(dat)

            save_folder = "../plots/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            df_ = pd.concat([df_one, df_M, df_fiveM, df_tenM])

            if model_name == "Linear Regression":
                subs_palette = sns.color_palette(palette="ocean")
                without_subs_palette = ["b"]
            elif model_name == "Logistic Regression":
                subs_palette = sns.color_palette(palette="gist_heat")
                without_subs_palette = ["r"]
            else:
                subs_palette = sns.color_palette(palette="gist_earth")
                without_subs_palette = ["g"]

            plt.figure()
            sns.lineplot(data=df_,
                         x="Coreset Size",
                         y="KL Divergence",
                         hue="Subsample Size",
                         style="Subsample Size",
                         markers=["D", "X", "s", "P"],
                         palette=subs_palette)
            sns.lineplot(data=df_N,
                         x="Coreset Size",
                         y="KL Divergence",
                         hue="Subsample Size",
                         style="Subsample Size",
                         markers=True,
                         palette=without_subs_palette)
            plt.title(f"{model_name}")
            plt.xlabel("Coreset Size")
            plt.ylabel("Two-Moment KL Divergence")
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder,
                        f"experiment_subsample_size_{kernel_name}_{model_name}_{timestamp}.svg"),
                        format="svg")

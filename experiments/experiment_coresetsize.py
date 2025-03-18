import logging
import os
from datetime import datetime

import experiment_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mcmc_sampler.kernels import HMC, MALA, SliceSampler

"""
This experiment illustrates the effect of the coreset size on the quality of the coreset.
Therefore each regression model (linear, logistic, poisson) is tested with the following coreset sizes M dependent on 
the size N of the full dataset:
M in {floor(N/1000), floor(N/500), floor(N/100), floor(N/50), floor(N/10)}
Five independent runs (Z=5) are started for each regression model for each coreset size for each MCMC choice. The mean 
of the two moment Kullback-Leibler divergence of these five runs is plotted for the corresponding model, coreset size,
and MCMC choice. This gives us three plots (one for each MCMC choice) where the x-axis is the coreset size, the y-axis
is the two moment KL divergence and a graph for each model.
"""

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"experiment_coreset_size_log_{timestamp}.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info(f"Start coreset size experiment")

    proportion = [1000, 500, 100, 50, 10]  #  Proportion of the corset of the full data set
    kernel = {"Slice Sampler": SliceSampler(),
              "HMC": HMC(),
              "MALA": MALA()}
    results = {"linear_regression": [],
               "logistic_regression": [],
               "poisson_regression": []}

    Z = 5

    for kernel_name, kernel in kernel.items():
        for model in results:
            for z in range(0, Z):
                kl_est_coreset_size = experiment_utils.run_experiment_csize(model_type=model,
                                                                            mcmc_kernel=kernel,
                                                                            frac=proportion)

                if not results[model]:
                    results[model] = kl_est_coreset_size
                else:
                    results[model] = [x + y for x, y in zip(results[model], kl_est_coreset_size)]

                logging.info(f"Kernel: {kernel_name}, "
                             f"Model: {model}, "
                             f"Z: {z}, "
                             f"results: {results[model]}")

        results = {model: [x / Z for x in values] for model, values in results.items()}

        x_labels = [f"N/{size}" for size in proportion]

        dat = {
            "Coreset Size": x_labels,
            "KL Divergence": results["linear_regression"],
            "Kernel": [f"{kernel_name}"] * len(proportion),
            "Model": ["Linear Regression"] * len(proportion)
        }
        df_lin_slice = pd.DataFrame(dat)

        dat["KL Divergence"] = results["logistic_regression"]
        dat["Kernel"] = [f"{kernel_name}"] * len(proportion)
        dat["Model"] = ["Logistic Regression"] * len(proportion)
        df_log_slice = pd.DataFrame(dat)

        dat["KL Divergence"] = results["poisson_regression"]
        dat["Kernel"] = [f"{kernel_name}"] * len(proportion)
        dat["Model"] = ["Poisson Regression"] * len(proportion)
        df_pois_slice = pd.DataFrame(dat)

        save_folder = "../plots/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        df_slice = pd.concat([df_lin_slice, df_log_slice, df_pois_slice])

        plt.figure()
        sns.lineplot(data=df_slice,
                     x="Coreset Size",
                     y="KL Divergence",
                     hue="Model",
                     style="Model",
                     markers=True,
                     palette=["b", "r", "g"])
        plt.title(kernel_name)
        plt.xlabel("Coreset Size")
        plt.ylabel("Two-Moment KL Divergence")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(save_folder,
                    f"experiment_coreset_size_{kernel_name}_{timestamp}.svg"),
                    format="svg")

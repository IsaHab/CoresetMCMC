import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mcmc_sampler.coreset_mcmc import CoresetMCMC
from mcmc_sampler.generators import CoresetGenerator
from mcmc_sampler.models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    PoissonRegressionModel,
)
from mcmc_sampler.unif_subsampling import UniformSubsampling


def load_data(model_type):
    data = None

    if model_type == "linear_regression":
        data = pd.read_csv("../data/linear_reg.csv", header=None)
    elif model_type == "logistic_regression":
        data = pd.read_csv("../data/logistic_reg.csv", header=None)
    elif model_type == "poisson_regression":
        data = pd.read_csv("../data/poisson_reg.csv", header=None)

    return data


def load_stan_results(model_type):
    stan_results = None

    if model_type == "linear_regression":
        with h5py.File("../stan_results/stan_lin_reg.jld", "r") as file:
            stan_results = file["data"][:]
    elif model_type == "logistic_regression":
        with h5py.File("../stan_results/stan_log_reg.jld", "r") as file:
            stan_results = file["data"][:]
    elif model_type == "poisson_regression":
        with h5py.File("../stan_results/stan_poisson_reg.jld", "r") as file:
            stan_results = file["data"][:]

    return stan_results


def initialize_model(model_type,
                     data):
    model = None

    # Intercept has to be included for linear and logistic regression
    if model_type == "linear_regression" or model_type == "logistic_regression":
        data_with_intercept = np.hstack(
            [np.ones((len(data), 1)), np.array(data)])
        data_list = [data_with_intercept[i, :] for i in
                     range(len(data_with_intercept))]

        N = len(data_list)
        d = data.shape[1] - 1

        if model_type == "linear_regression":
            model = LinearRegressionModel(
                N=N,
                dataset=data_list,
                datamat=data_with_intercept,
                d=d,
                sigma_prior=1.0,
                mu_prior=np.zeros(data_list[0].shape[0])
            )
        elif model_type == "logistic_regression":
            model = LogisticRegressionModel(
                N=N,
                dataset=data_list,
                datamat=data_with_intercept,
                d=d
            )
    elif model_type == "poisson_regression":
        data_matrix = data.to_numpy()
        # Bring data in correct form [1, x1, ..., xd, y]
        rearranged_data = np.hstack([data_matrix[:, -2: -1],
                                     data_matrix[:, :-2],
                                     data_matrix[:, -1:]])
        data_list = [rearranged_data[i, :] for i in
                     range(rearranged_data.shape[0])]
        N = len(data_list)
        d = data.shape[1] - 1

        model = PoissonRegressionModel(
            N=N,
            dataset=data_list,
            datamat=rearranged_data,
            d=d,
            sigma_prior=1.0
        )
    return model


def initialize_sampler(model,
                       mcmc_kernel,
                       alpha,
                       K,
                       train_iter,
                       subsampling):
    subs = subsampling[0]  # subsampling Y/N
    if subs:
        S = subsampling[1]  # subsample size
    else:
        S = model.N

    coreset_mcmc = CoresetMCMC(
        kernel=mcmc_kernel,
        K=K,
        alpha=lambda t: float(alpha),
        train_iter=train_iter,
        S=S
    )

    return coreset_mcmc


def get_samples(model,
                mcmc_kernel,
                n_samples,
                coreset_size,
                alpha,
                K,
                train_iter,
                subsample):
    mcmc_sampler = initialize_sampler(model=model,
                                      mcmc_kernel=mcmc_kernel,
                                      alpha=alpha,
                                      K=K,
                                      train_iter=train_iter,
                                      subsampling=subsample)
    results = mcmc_sampler.sample(model=model,
                                  coreset=CoresetGenerator(
                                      subsamples_num=coreset_size),
                                  n_samples=n_samples)
    thetas = results.samples
    final_samples = np.vstack(thetas[-n_samples:])

    return final_samples


def calculate_metric(model_type,
                     final_samples):
    stan_results = load_stan_results(model_type=model_type)

    stan_mean = np.mean(stan_results.T, axis=0)
    stan_cov = np.cov(stan_results)
    method_mean = np.mean(final_samples, axis=0)
    method_cov = np.cov(final_samples.T)

    kl_est = kl_gaussian(mu_q=method_mean,
                         sigma_q=method_cov,
                         mu_p=stan_mean,
                         sigma_p=stan_cov)
    return kl_est


def kl_gaussian(mu_q: NDArray,
                sigma_q: NDArray,
                mu_p: NDArray,
                sigma_p: NDArray) -> float:
    d = len(mu_q)
    try:
        sign_q, logdet_q = np.linalg.slogdet(sigma_q)
        if sign_q <= 0:
            logdet_q = np.log(1e-20)
    except np.linalg.LinAlgError:
        logdet_q = np.log(1e-20)

    sign_p, logdet_p = np.linalg.slogdet(sigma_p)
    sigma_p_inv = np.linalg.inv(sigma_p)
    trace_term = np.trace(sigma_p_inv @ sigma_q)
    diff = mu_p - mu_q
    quad_term = diff.T @ sigma_p_inv @ diff

    kl = 0.5 * (logdet_p - logdet_q - d + trace_term + quad_term)

    return float(kl)


def run(model_type,
        model,
        mcmc_kernel,
        n_samples,
        coreset_size,
        alpha,
        K,
        train_iter,
        subsample):
    final_samples = get_samples(model=model,
                                mcmc_kernel=mcmc_kernel,
                                n_samples=n_samples,
                                coreset_size=coreset_size,
                                alpha=alpha,
                                K=K,
                                train_iter=train_iter,
                                subsample=subsample)
    kl_est = calculate_metric(model_type=model_type,
                              final_samples=final_samples)
    return kl_est


def run_experiment_csize(model_type,
                         mcmc_kernel,
                         frac,
                         subsampling=None):
    if subsampling is None:
        subsampling = [[False]]

    num = len(frac)
    alpha = 0.01
    train_iter = 25000
    K = 2
    n_samples = 10000
    subsample = subsampling[0]

    kl_est_coreset_sizes = []

    data = load_data(model_type=model_type)
    model = initialize_model(model_type=model_type,
                             data=data)

    for i in range(0, num):
        kl_est = run(model_type=model_type,
                     model=model,
                     mcmc_kernel=mcmc_kernel,
                     n_samples=n_samples,
                     coreset_size=int(np.floor(model.N / frac[i])),
                     alpha=alpha,
                     K=K,
                     train_iter=train_iter,
                     subsample=subsample)
        kl_est_coreset_sizes.append(kl_est)

    return kl_est_coreset_sizes


def run_experiment_subssize(model_type,
                            mcmc_kernel,
                            frac,
                            subsampling):
    num = len(frac)
    alpha = 0.01
    train_iter = 25000
    K = 2
    n_samples = 10000

    kl_est_coreset_size = []

    data = load_data(model_type=model_type)
    model = initialize_model(model_type=model_type,
                             data=data)

    for i in range(0, num):
        kl_est = run(model_type=model_type,
                     model=model,
                     mcmc_kernel=mcmc_kernel,
                     n_samples=n_samples,
                     coreset_size=int(np.floor(model.N / frac[i])),
                     alpha=alpha,
                     K=K,
                     train_iter=train_iter,
                     subsample=[True,
                                subsampling * int(np.floor(model.N / frac[i]))])
        kl_est_coreset_size.append(kl_est)

    return kl_est_coreset_size


def run_unif(model_type,
             model,
             mcmc_kernel,
             coreset_size,
             n_samples,
             T):
    kl_est_runs = []

    unif = UniformSubsampling(kernel=mcmc_kernel,
                              T=T)
    results = unif.sample(model=model,
                          generator=CoresetGenerator(
                              subsamples_num=coreset_size))

    thetas = results.samples
    final_samples = np.vstack(thetas[-n_samples:])

    kl_est = calculate_metric(model_type=model_type,
                              final_samples=final_samples)
    kl_est_runs.append(kl_est)

    mean_kl_est = np.mean(kl_est_runs)
    return mean_kl_est


def run_experiment_unif(model_type,
                        mcmc_kernel,
                        frac):
    num = len(frac)
    T = 35000
    n_samples = 10000

    kl_est_coreset_size = []

    data = load_data(model_type=model_type)
    model = initialize_model(model_type=model_type,
                             data=data)

    for i in range(0, num):
        mean_kl_est = run_unif(model_type=model_type,
                               model=model,
                               mcmc_kernel=mcmc_kernel,
                               coreset_size=int(np.floor(model.N / frac[i])),
                               n_samples=n_samples,
                               T=T)
        kl_est_coreset_size.append(mean_kl_est)

    return kl_est_coreset_size

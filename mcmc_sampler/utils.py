import numpy as np
from numpy.typing import NDArray

from mcmc_sampler import MetaState
from mcmc_sampler.models import AbstractModel


def centered_logliks(meta_state: MetaState,
                     model: AbstractModel,
                     subsample_data: NDArray) -> NDArray:

    K = len(meta_state.states)
    temp = np.zeros((subsample_data.shape[0], K))

    for k in range(K):
        result = model.log_likelihood_array(state=meta_state.states[k],
                                            subset=subsample_data)
        temp[:, k] = result

    temp -= np.mean(temp, axis=1, keepdims=True)
    return temp


def centered_logliks_sum(meta_state: MetaState,
                         model: AbstractModel,
                         subsample_data: NDArray) -> NDArray:

    temp = centered_logliks(meta_state=meta_state,
                            model=model,
                            subsample_data=subsample_data)

    return (model.N / subsample_data.shape[0]) * np.sum(temp, axis=0)


def log_logistic(a):
    """ Numerically more stable method to calculate -log(1+exp(-a))"""
    return -np.logaddexp(0, -a)

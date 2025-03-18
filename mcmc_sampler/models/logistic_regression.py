from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

import mcmc_sampler.utils as utils
from mcmc_sampler import State
from mcmc_sampler.models import AbstractModel


class LogisticRegressionModel(AbstractModel):

    def __init__(self,
                 N: int,
                 dataset: List[NDArray],
                 datamat: NDArray,
                 d: int):
        super().__init__(N,
                         dataset,
                         datamat,
                         d)

    def create_initial_state(self,
                             rng: np.random.Generator,
                             init_val: Optional[NDArray] = None) -> State:
        if init_val is None:
            theta0 = np.zeros(len(self.dataset[0]) - 1)
        else:
            theta0 = init_val
        return State(theta=theta0, rng=rng)

    def log_prior(self,
                  theta: NDArray) -> float:
        return -np.sum(np.log1p(theta ** 2))

    def log_likelihood(self,
                       x,
                       state: State) -> float:
        return (x[-1] * utils.log_logistic(np.dot(state.theta, x[:-1])) + (1 - x[-1])
                * utils.log_logistic(-np.dot(state.theta, x[:-1])))

    def log_likelihood_array(self,
                             state: State,
                             subset: NDArray) -> NDArray:
        xs = subset[:, :-1]
        ys = subset[:, -1]

        log_lik_pos = ys * utils.log_logistic(xs @ state.theta)
        log_lik_neg = (1 - ys) * utils.log_logistic(-xs @ state.theta)

        return log_lik_pos + log_lik_neg

    def grad_log_prior(self,
                       state: State) -> NDArray:
        return -2 * state.theta / (1 + state.theta ** 2)

    def grad_log_likelihood_array(self,
                                  state: State,
                                  generator) -> np.ndarray:
        coreset = generator.sub_dataset

        lp = np.dot(coreset[:, :-1], state.theta)
        lp = np.clip(lp, -500, 500)
        logistic = 1 / (1 + np.exp(-lp))
        grad_theta = (coreset[:, -1] - logistic)[:, np.newaxis] * coreset[:, :-1]

        return grad_theta

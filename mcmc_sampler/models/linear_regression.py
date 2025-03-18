from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from mcmc_sampler import State
from mcmc_sampler.models import AbstractModel


class LinearRegressionModel(AbstractModel):
    def __init__(self,
                 N: int,
                 dataset: List[NDArray],
                 datamat: NDArray,
                 d: int,
                 sigma_prior: float,
                 mu_prior: NDArray):
        super().__init__(N,
                         dataset,
                         datamat,
                         d)
        self.sigma_prior = sigma_prior
        self.mu_prior = mu_prior

    def create_initial_state(self,
                             rng: np.random.Generator,
                             init_val: Optional[NDArray] = None) -> State:
        if init_val is None:
            theta0 = np.ones(len(self.dataset[0])) * 10
            theta0[:-1] = 0.0
        else:
            theta0 = init_val
        return State(theta=theta0, rng=rng)

    def log_prior(self,
                  theta: NDArray) -> float:
        diff = theta - self.mu_prior
        return -0.5 * np.dot(diff, diff) / (self.sigma_prior ** 2)

    def grad_log_prior(self,
                       state: State) -> NDArray:
        return -(state.theta - self.mu_prior) / (self.sigma_prior ** 2)

    def log_likelihood(self,
                       x,
                       state: State) -> float:
        return (-0.5 * state.theta[-1] - (1 / (2 * np.exp(state.theta[-1])))
                * ((np.dot(state.theta[:-1], x[:-1]) - x[-1]) ** 2))

    def log_likelihood_array(self,
                             state: State,
                             subset: NDArray) -> NDArray:
        xs = subset[:, :-1]
        ys = subset[:, -1]

        pred = np.dot(xs, state.theta[:self.d + 1])
        residuals = (pred - ys) ** 2

        theta_val = np.clip(state.theta[-1], -500, 500)
        return -0.5 * (theta_val + residuals / np.exp(theta_val))

    def grad_log_likelihood_array(self,
                                  state: State,
                                  generator) -> NDArray:
        coreset = generator.sub_dataset
        max_theta = 700
        state.theta[-1] = np.clip(state.theta[-1], -max_theta, max_theta)

        diff = coreset[:, -1] - np.dot(coreset[:, :-1], state.theta[:-1])
        part1 = diff[:, np.newaxis] * coreset[:, :-1] / np.exp(state.theta[-1])
        part2 = 0.5 * np.exp(-state.theta[-1]) * (diff ** 2) - 0.5
        grad_theta = np.hstack([part1, part2[:, np.newaxis]])

        return grad_theta

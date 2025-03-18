from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from mcmc_sampler import State, utils
from mcmc_sampler.models import AbstractModel


class PoissonRegressionModel(AbstractModel):
    def __init__(self,
                 N: int,
                 dataset: List[NDArray],
                 datamat: NDArray,
                 d: int,
                 sigma_prior: float):
        super().__init__(N,
                         dataset,
                         datamat,
                         d)
        self.sigma_prior = sigma_prior

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
        return -0.5 * np.dot(theta, theta) / (self.sigma_prior ** 2)

    def log_likelihood(self,
                       x,
                       state: State) -> float:
        return (x[-1] * np.log(-utils.log_logistic(-np.dot(x[:-1], state.theta)))
                + utils.log_logistic(-np.dot(x[:-1], state.theta)))

    def log_likelihood_array(self,
                             state: State,
                             subset: NDArray) -> NDArray:
        xs = subset[:, :-1]
        ys = subset[:, -1]

        prods = np.dot(xs, state.theta)
        prods = np.clip(prods, -500, 500)

        return ys * np.log(-utils.log_logistic(-prods)) + utils.log_logistic(-prods)

    def grad_log_prior(self,
                       state: State) -> NDArray:
        return -state.theta / self.sigma_prior ** 2

    def grad_log_likelihood_array(self,
                                  state: State,
                                  generator) -> np.ndarray:
        coreset = generator.sub_dataset

        lp = np.dot(coreset[:, :-1], state.theta)
        lp = np.clip(lp, -500, 500)
        logistic = 1 / (1 + np.exp(-lp))
        log_term = coreset[:, -1] / (-utils.log_logistic(-lp))
        grad_theta = ((log_term - 1) * logistic)[:, np.newaxis] * coreset[:, :-1]

        return grad_theta

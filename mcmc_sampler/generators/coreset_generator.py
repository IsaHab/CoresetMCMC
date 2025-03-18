from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mcmc_sampler import State
from mcmc_sampler.generators import AbstractGenerator


class CoresetGenerator(AbstractGenerator):

    def init_generator(self,
                       model) -> None:
        if self.inds is None:
            self.inds = np.random.choice(np.arange(model.N),
                                         size=self.subsamples_num,
                                         replace=False)

        self.sub_dataset = model.datamat[self.inds]
        self.sub_xp = self.sub_dataset.T
        self.weights = (model.N / self.subsamples_num) * np.ones(self.subsamples_num)

    def update_generator(self,
                         model,
                         theta_prime: Optional[NDArray] = None,
                         mu0: Optional[float] = None,
                         inds: Optional[NDArray] = None) -> float:
        return 0.0

    def log_likelihood_weighted(self,
                                state: State,
                                model) -> float:
        logliks = model.log_likelihood_array(state, self.sub_dataset)
        return np.sum(logliks * self.weights)

    def grad_log_likelihood_diff(self,
                                 state: State,
                                 model) -> float:
        grad_logliks = model.grad_log_likelihood_array(state, self)
        return np.sum(grad_logliks * self.weights[:, np.newaxis], axis=0)

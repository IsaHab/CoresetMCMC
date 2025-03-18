from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import NDArray

from mcmc_sampler import State


class AbstractGenerator(ABC):
    def __init__(self, subsamples_num: int):
        self.subsamples_num = subsamples_num
        self.inds: Optional[NDArray] = None
        self.sub_dataset: Optional[NDArray] = None
        self.sub_xp: Optional[NDArray] = None
        self.weights: Optional[NDArray] = None

    @abstractmethod
    def update_generator(self,
                         model,
                         theta_prime: Optional[NDArray] = None,
                         mu0: Optional[float] = None,
                         inds: Optional[NDArray] = None) -> float:
        pass

    @abstractmethod
    def log_likelihood_weighted(self,
                                state: State,
                                model) -> float:
        pass

    @abstractmethod
    def grad_log_likelihood_diff(self,
                                 state: State,
                                 model) -> float:
        pass

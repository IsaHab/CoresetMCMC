from abc import ABC, abstractmethod
from typing import List

from numpy.typing import NDArray

from mcmc_sampler import State
from mcmc_sampler.generators import AbstractGenerator


class AbstractModel(ABC):

    def __init__(self,
                 N: int,
                 dataset: List[NDArray],  # [1, x1, ..., xN, y]
                 datamat: NDArray,
                 d: int):
        self.N = N
        self.dataset = dataset
        self.datamat = datamat
        self.d = d

    @abstractmethod
    def create_initial_state(self,
                             new_rng,
                             init_val):
        pass

    @abstractmethod
    def log_prior(self,
                  theta: NDArray) -> float:
        pass

    @abstractmethod
    def log_likelihood(self,
                       x,
                       state: State) -> float:
        pass

    @abstractmethod
    def log_likelihood_array(self,
                             state,
                             subset):
        pass

    @abstractmethod
    def grad_log_prior(self,
                       state: State) -> NDArray:
        pass

    @abstractmethod
    def grad_log_likelihood_array(self,
                                  state: State,
                                  generator) -> NDArray:
        pass

    def log_potential(self,
                      state: State,
                      generator: AbstractGenerator) -> float:
        return self.log_prior(state.theta) + generator.log_likelihood_weighted(
            state, self)

    def grad_log_potential(self,
                           state: State,
                           generator: AbstractGenerator) -> NDArray:
        return self.grad_log_prior(state) + generator.grad_log_likelihood_diff(state, self)
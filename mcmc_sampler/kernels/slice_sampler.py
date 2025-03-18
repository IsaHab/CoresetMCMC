from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import expon

from mcmc_sampler import AbstractKernel, State
from mcmc_sampler.generators import AbstractGenerator
from mcmc_sampler.models import AbstractModel


class SliceSampler(AbstractKernel):
    """
        Hit-and-run stepping-out shrinkage slice sampler
        Krzysztof Latuszynski and Daniel Rudolf. “Convergence of hybrid slice sampling via spectral gap”.
        In: Advances in Applied Probability 56.4 (2024), pp. 1440–1466.
    """

    def __init__(self, w: float = 10.0):
        self.w = w

    def step(self,
             state: State,
             model: AbstractModel,
             generator: AbstractGenerator,
             iteration: int) -> NDArray:

        cached_lp = model.log_potential(state=state,
                                        generator=generator)
        self._slice_sample(state=state,
                           model=model,
                           generator=generator,
                           cached_lp=cached_lp,
                           theta=state.theta.copy())
        return state.theta

    def _slice_sample(self,
                      state: State,
                      model: AbstractModel,
                      generator: AbstractGenerator,
                      cached_lp: float,
                      theta: NDArray) -> float:

        z = cached_lp - expon.rvs(size=1, random_state=state.rng)[0]
        search_dir = state.rng.random(len(theta))
        search_dir = search_dir / np.linalg.norm(search_dir)

        L, R = self.stepping_out(state=state,
                                 model=model,
                                 generator=generator,
                                 z=z,
                                 theta=theta,
                                 search_dir=search_dir)
        cached_lp = self.slice_shrink(state=state,
                                      model=model,
                                      generator=generator,
                                      z=z,
                                      L=L,
                                      R=R,
                                      theta=theta,
                                      search_dir=search_dir)
        return cached_lp

    def stepping_out(self,
                     state: State,
                     model: AbstractModel,
                     generator: AbstractGenerator,
                     z: float,
                     theta: NDArray,
                     search_dir: NDArray) -> Tuple[float, float]:
        old_position = theta.copy()
        L, R = self.initialize_slice_endpoints(rng=state.rng)

        state.theta = old_position - L * search_dir
        potent_L = model.log_potential(state=state,
                                       generator=generator)
        state.theta = old_position + R * search_dir
        potent_R = model.log_potential(state=state,
                                       generator=generator)

        while (z < potent_L):
            L = L + self.w
            state.theta = old_position - L * search_dir
            potent_L = model.log_potential(state=state,
                                           generator=generator)

        while (z < potent_R):
            R = R + self.w
            state.theta = old_position + R * search_dir
            potent_R = model.log_potential(state=state,
                                           generator=generator)

        state.theta = old_position
        return L, R

    def initialize_slice_endpoints(self,
                                   rng: np.random.Generator) -> Tuple[float, float]:
        L = self.w * rng.random()
        R = self.w - L
        return L, R

    def slice_shrink(self,
                     state: State,
                     model: AbstractModel,
                     generator: AbstractGenerator,
                     z: float,
                     L: float,
                     R: float,
                     theta: NDArray,
                     search_dir: NDArray) -> float:
        old_position = theta.copy()
        Lbar = -L
        Rbar = R

        while True:
            s = Lbar + state.rng.random() * (Rbar - Lbar)
            new_position = old_position + s * search_dir
            state.theta = new_position.copy()
            new_lp = model.log_potential(state=state,
                                         generator=generator)

            if z < new_lp:
                state.theta = new_position.copy()
                return new_lp

            if s < 0:
                Lbar = s
            else:
                Rbar = s

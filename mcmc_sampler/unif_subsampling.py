from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from mcmc_sampler import AbstractKernel, MetaState, SamplingResults
from mcmc_sampler.generators import CoresetGenerator
from mcmc_sampler.models import AbstractModel


class UniformSubsampling:
    def __init__(self,
                 kernel: AbstractKernel,
                 T: int = 35000):
        self.kernel = kernel
        self.T = T

    def sample(self,
               model: AbstractModel,
               generator: CoresetGenerator,
               init_val: Optional[NDArray] = None) -> SamplingResults:
        meta_state = MetaState()
        samples = []
        weights = []

        new_rng = np.random.default_rng()
        meta_state.states.append(model.create_initial_state(new_rng, init_val))

        for i in tqdm(range(self.T), desc="Step"):
            if i == 0:
                generator.init_generator(model=model)

            self.kernel.step(state=meta_state.states[0],
                             model=model,
                             generator=generator,
                             iteration=i)

            for state in meta_state.states:
                samples.append(state.theta.copy())
                weights.append(generator.weights.copy())

        return SamplingResults(
            samples=samples,
            weights=weights
        )


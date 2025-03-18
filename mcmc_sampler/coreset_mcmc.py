from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from mcmc_sampler import AbstractKernel, MetaState, SamplingResults
from mcmc_sampler.generators import CoresetGenerator
from mcmc_sampler.models import AbstractModel


class CoresetMCMC:
    def __init__(self,
                 kernel: AbstractKernel,
                 K: int = 2,
                 train_iter: int = 25000,
                 alpha: Callable[[int], float] = lambda t: 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-3,
                 t: int = 0,
                 m: Optional[NDArray] = None,
                 v: Optional[NDArray] = None,
                 subsample_data: Optional[NDArray] = None,
                 S: int = 1000):
        self.kernel = kernel
        self.K = K
        self.train_iter = train_iter
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t
        self.m = m if m is not None else np.zeros(S)
        self.v = v if v is not None else np.zeros(S)
        self.subsample_data = subsample_data
        self.S = S

    def sample(self,
               model: AbstractModel,
               coreset: CoresetGenerator,
               n_samples: int,
               init_val: Optional[NDArray] = None) -> SamplingResults:
        meta_state = MetaState()
        samples = []
        weights = []
        for _ in range(self.K):
            new_rng = np.random.default_rng()
            meta_state.states.append(
                model.create_initial_state(new_rng, init_val))

        total_steps = self.train_iter + int(np.ceil(n_samples / self.K))

        for i in tqdm(range(total_steps), desc="Step"):
            self.step(meta_state, model, coreset, i)
            for state in meta_state.states:
                samples.append(state.theta.copy())
            weights.append(coreset.weights.copy())

        return SamplingResults(
            samples=samples,
            weights=weights
        )

    def step(self,
             meta_state: MetaState,
             model: AbstractModel,
             coreset: CoresetGenerator,
             iteration: int) -> None:
        # Initialize on first iteration
        if iteration == 0:
            coreset.init_generator(model=model)
            self.m = np.zeros(coreset.subsamples_num)  # momentum
            self.v = np.zeros(coreset.subsamples_num)  # velocity

        # Subsampling step
        if self.subsample_data is None:
            indices = np.random.choice(np.arange(model.N),
                                       size=self.S,
                                       replace=False)
            self.subsample_data = model.datamat[indices]

        for i in range(self.K):
            self.kernel.step(state=meta_state.states[i],
                             model=model,
                             generator=coreset,
                             iteration=iteration)

        # Update coreset weights via ADAM
        if iteration <= self.train_iter:
            self.update_weights(meta_state=meta_state,
                                model=model,
                                coreset=coreset)

    def update_weights(self,
                       meta_state: MetaState,
                       model: AbstractModel,
                       coreset: CoresetGenerator) -> None:
        """
            Update coreset weights using ADAM optimizer.
            See pseudocode in Kingma, Diederik P., and Jimmy Lei Ba.
            "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION."
        """
        self.t += 1
        g = self.est_gradient(meta_state, model, coreset, self.subsample_data)

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        coreset.weights -= self.alpha(self.t) * m_hat / (np.sqrt(v_hat) + self.epsilon)
        coreset.weights = np.maximum(0., coreset.weights)

    def est_gradient(self,
                     meta_state: MetaState,
                     model: AbstractModel,
                     coreset: CoresetGenerator,
                     subsample_data: NDArray) -> NDArray:
        from mcmc_sampler.utils import centered_logliks, centered_logliks_sum

        g = centered_logliks(meta_state=meta_state,
                             model=model,
                             subsample_data=coreset.sub_dataset)
        center_sum = centered_logliks_sum(meta_state=meta_state,
                                          model=model,
                                          subsample_data=subsample_data)
        h = center_sum - np.dot(g.T, coreset.weights)
        grad = -np.dot(g, h) / (len(meta_state.states) - 1)

        return grad

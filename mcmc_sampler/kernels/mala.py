import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from mcmc_sampler import AbstractKernel, State
from mcmc_sampler.generators import AbstractGenerator
from mcmc_sampler.models import AbstractModel


class MALA(AbstractKernel):
    """
        MALA - Metropolis adjusted Langevin Algorithm
        T Xifara et al. “Langevin diffusions and the Metropolis-adjusted Langevin algorithm”.
        In: Statistics & Probability Letters 91.C (2014), pp. 14–19.
    """

    def __init__(self,
                 epsilon: float = 1e-3,
                 target_accept_rate: float = 0.574,
                 adaptation_window: int = 50):
        self.epsilon = epsilon
        self.target_accept_rate = target_accept_rate
        self.adaptation_window = adaptation_window
        self.acceptance_history = []
        self._adaptation_counter = 0

    def step(self,
             state: State,
             model: AbstractModel,
             generator: AbstractGenerator,
             iteration: int) -> NDArray:

        z = state.rng.normal(size=len(state.theta))
        state_current = state.copy()
        grad_U_current = model.grad_log_potential(state=state_current,
                                                  generator=generator)

        state_proposed = state.copy()
        state_proposed.theta = state_current.theta + self.epsilon * grad_U_current + np.sqrt(2 * self.epsilon) * z
        U_current = model.log_potential(state=state_current,
                                        generator=generator)
        U_proposed = model.log_potential(state=state_proposed,
                                         generator=generator)

        grad_U_proposed = model.grad_log_potential(state=state_proposed,
                                                   generator=generator)

        q_forward = multivariate_normal.logpdf(state_proposed.theta,
                                               mean=state.theta + self.epsilon * grad_U_current,
                                               cov=2 * self.epsilon * np.eye(len(state.theta)))
        q_backward = multivariate_normal.logpdf(state.theta,
                                                mean=state_proposed.theta + self.epsilon * grad_U_proposed,
                                                cov=2 * self.epsilon * np.eye(len(state.theta)))

        log_acceptance_ratio = (U_proposed + q_backward) - (U_current + q_forward)
        accept_prob = np.exp(np.clip(log_acceptance_ratio, -700, 700))
        accepted = state.rng.uniform() < min(1, accept_prob)

        if accepted:
            state.theta = state_proposed.theta

        self.acceptance_history.append(accepted)
        self._adaptation_counter += 1

        if self._adaptation_counter >= self.adaptation_window:
            recent_accept_rate = np.mean(self.acceptance_history[-self.adaptation_window:])

            if recent_accept_rate < self.target_accept_rate:
                self.epsilon *= 0.95
            else:
                self.epsilon *= 1.05

            self.epsilon = np.clip(self.epsilon, 1e-7, 0.1)
            self._adaptation_counter = 0

        return state.theta

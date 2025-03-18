import numpy as np
from numpy.typing import NDArray

from mcmc_sampler import AbstractKernel, State
from mcmc_sampler.generators import AbstractGenerator
from mcmc_sampler.models import AbstractModel


class HMC(AbstractKernel):
    """
       HMC - Hamiltonian Monte Carlo
       Radford M Neal. “MCMC Using Hamiltonian Dynamics”.
       In: Handbook ofMarkov Chain Monte Carlo. Chapman and Hall/CRC, 2011, pp. 113–162.
    """
    def __init__(self,
                 epsilon: float = 1e-3,
                 target_accept_rate: float = 0.65,
                 adaptation_window: int = 50,
                 L: int = 20):
        self.epsilon = epsilon
        self.target_accept_rate = target_accept_rate
        self.adaptation_window = adaptation_window
        self.acceptance_history = []
        self.L = L
        self._adaptation_counter = 0

    def step(self,
             state: State,
             model: AbstractModel,
             generator: AbstractGenerator,
             iteration: int) -> NDArray:
        r = state.rng.standard_normal(len(state.theta))
        U_current = -model.log_potential(state=state,
                                         generator=generator)
        H_current = U_current + 0.5 * np.dot(r, r)

        state_proposed = state.copy()
        r_temp = r.copy()

        grad = model.grad_log_potential(state=state_proposed,
                                        generator=generator)
        for _ in range(self.L):
            r_temp += 0.5 * self.epsilon * grad
            state_proposed.theta += self.epsilon * r_temp
            grad = model.grad_log_potential(state=state_proposed,
                                            generator=generator)
            r_temp += 0.5 * self.epsilon * grad

        U_proposed = -model.log_potential(state=state_proposed,
                                          generator=generator)
        H_proposed = U_proposed + 0.5 * np.dot(r_temp, r_temp)

        H_diff = np.clip(H_current - H_proposed, -700, 700)
        accept_prob = np.exp(H_diff)
        accepted = state.rng.random() < min(1, accept_prob)

        if accepted:
            state.theta = state_proposed.theta.copy()

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

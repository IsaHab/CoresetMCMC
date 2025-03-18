from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class SamplingResults:
    samples: List[NDArray]
    weights: Optional[List[NDArray]] = None


@dataclass
class State:
    theta: Union[NDArray, float]
    rng: np.random.Generator
    weight: Optional[float] = None

    def copy(self) -> 'State':
        return State(
            theta=np.copy(self.theta) if isinstance(self.theta,
                                                    np.ndarray) else self.theta,
            rng=np.random.Generator(self.rng.bit_generator),
            weight=self.weight
        )


@dataclass
class MetaState:
    states: List[State] = field(default_factory=list)

    def copy(self) -> 'MetaState':
        return MetaState(states=[state.copy() for state in self.states])

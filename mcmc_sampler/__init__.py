from abc import ABC, abstractmethod

from .dataclasses import MetaState, SamplingResults, State


class AbstractKernel(ABC):

    @abstractmethod
    def step(self,
             state,
             model,
             generator,
             iteration):
        pass


# Module exports
__all__ = [
    'AbstractKernel',
    'State',
    'MetaState',
    'SamplingResults'
]

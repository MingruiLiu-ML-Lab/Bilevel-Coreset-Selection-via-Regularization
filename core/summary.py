import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from core.utils import compute_and_flatten_example_grads, flatten_example_grads
from core.mode_connectivity import get_ti
import os
from .utils import load_model
import time



class Summarizer(ABC):

    def __init__(self, rs=None):
        super().__init__()
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs

    @abstractmethod
    def build_summary(self, X, y, size, **kwargs):
        pass

    def factory(type, rs):
        if type == 'uniform': return UniformSummarizer(rs)
        raise TypeError('Unkown summarizer type ' + type)

    factory = staticmethod(factory)


class UniformSummarizer(Summarizer):

    def build_summary(self, X, y, size, **kwargs):
        n = X.shape[0]
        inds = self.rs.choice(n, size, replace=False)

        return inds


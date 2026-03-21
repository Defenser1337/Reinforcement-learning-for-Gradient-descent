import numpy as np
from typing import Optional
from src.optimization.function import Function
from src.optimization.convex_function import ConvexFunction
from src.utils.prng import get_rng

class ConvexFunctionWithNoise(Function):
    """
    Represents a convex funtion with noise in form of: 
        f(x) = C(x) + N(x)

    Where:
        C(x) - convex function
        N(x) - high frequency sine wave A * (1 + sin(w * (x1 + x2 + ... + xn)))

    Parameters:
        in_features (int): dimension of the input vector
    """

    def __init__ (self, in_features : int,
                  amplitude : float = 1.0,
                  frequency : float = 5.0,
                  random_state : Optional[int] = None,
                  max_absolute_value : float = 1.0,
                  tol: float = 1e-10) -> None:
        super().__init__(in_features)
        self._tol = tol
        self._max_absolute_value = max_absolute_value
        self._eps = 1e-12
        self._amplitude = amplitude
        self._frequency = frequency 

        self._rng = get_rng(seed=random_state, location_name="convex_function_with_noise_class")

        self._convex_function_seed = self._rng.integers(0, 2**31 - 1)

        self._counvex_function = ConvexFunction(in_features=self.in_features, random_state=self._convex_function_seed, max_absolute_value=self.max_absolute_value)
    
    def __call__(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Vector dimension is incorrect")
        return self._counvex_function(x) + self.amplitude * (1 + np.sin(self.frequency * np.sum(x)))
    
    def get_gradient(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Vector dimension is incorrect")
        return self._counvex_function.get_gradient(x) + self.frequency * self.amplitude * np.full(self.in_features, np.cos(self.frequency * np.sum(x)))

    @property
    def max_absolute_value(self):
        return self._max_absolute_value

    @property
    def amplitude(self):
        return self._amplitude
    
    @property
    def frequency(self):
        return self._frequency
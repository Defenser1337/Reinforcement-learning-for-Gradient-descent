import numpy as np
from typing import Optional
from src.optimization.optimization_functions.function import Function
from src.optimization.optimization_functions.convex_function import ConvexFunction
from src.utils.prng import get_rng

AMPLITUDE_LOWER_BOUND = 0.1
AMPLITUDE_UPPER_BOUND = 5.0
FREQUENCY_LOWER_BOUND = 1.0
FREQUENCY_UPPER_BOUND = 20.0

class ConvexFunctionWithNoise(Function):
    """
    Represents a convex function with noise in form of:
        f(x) = C(x) + N(x)
    Where:
        C(x) - convex function
        N(x) = (A / N) * sum(1 + sin(w * x_i + phi_i)) for i in 1..N
        A    - noise amplitude
        w    - noise frequency
        phi_i - random phase shift per coordinate
    Parameters:
        in_features (int)  : dimension of the input vector
        scale (float)      : scale parameter for the convex part
        amplitude (float)  : amplitude of the noise
        frequency (float)  : frequency of the noise
        seed (int)         : generation seed
    """
    def __init__(self, in_features: Optional[int] = None,
                       scale: Optional[float] = None,
                       amplitude: Optional[float] = None,
                       frequency: Optional[float] = None,
                       seed: Optional[int] = None):
        self._seed = seed
        self._rng = get_rng(seed=seed, location_name="convex_function_with_noise_class")

        if amplitude is not None and (amplitude < AMPLITUDE_LOWER_BOUND or amplitude > AMPLITUDE_UPPER_BOUND):
            raise ValueError(f"The specified parameter (amplitude) is out of bounds [{AMPLITUDE_LOWER_BOUND}, {AMPLITUDE_UPPER_BOUND}].")
        if frequency is not None and (frequency < FREQUENCY_LOWER_BOUND or frequency > FREQUENCY_UPPER_BOUND):
            raise ValueError(f"The specified parameter (frequency) is out of bounds [{FREQUENCY_LOWER_BOUND}, {FREQUENCY_UPPER_BOUND}].")

        self._amplitude = amplitude if amplitude is not None else self._rng.uniform(AMPLITUDE_LOWER_BOUND, AMPLITUDE_UPPER_BOUND)
        self._frequency = frequency if frequency is not None else self._rng.uniform(FREQUENCY_LOWER_BOUND, FREQUENCY_UPPER_BOUND)

        convex_seed = self._rng.integers(0, 2**31 - 1)
        self._convex_function = ConvexFunction(in_features=in_features, scale=scale, seed=convex_seed)

        super().__init__(in_features=self._convex_function.in_features)
        self._scale = self._convex_function.scale

        self._phases = self._rng.uniform(0, 2 * np.pi, size=self.in_features)

    def __call__(self, x: np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
            raise ValueError("Input vector dim is incorrect.")
        noise = (self._amplitude / self.in_features) * np.sum(1 + np.sin(self._frequency * x + self._phases))
        return self._convex_function(x) + noise

    def get_gradient(self, x: np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
            raise ValueError("Input vector dim is incorrect.")
        noise_grad = (self._amplitude / self.in_features) * self._frequency * np.cos(self._frequency * x + self._phases)
        return self._convex_function.get_gradient(x) + noise_grad

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def frequency(self):
        return self._frequency

    @property
    def phases(self):
        return self._phases.copy()

    @property
    def seed(self):
        return self._seed
    
    @property
    def scale(self):
        return self._scale

    @property
    def convex_function(self):
        return self._convex_function
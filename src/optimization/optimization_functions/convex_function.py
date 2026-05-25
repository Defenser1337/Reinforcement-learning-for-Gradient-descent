import numpy as np
from typing import Optional
from src.optimization.optimization_functions.function import Function
from src.utils.prng import get_rng

SCALE_LOWER_BOUND = 0.01
SCALE_UPPER_BOUND = 100
OPT_POINT_BOX_LOWER_BOUND = -5
OPT_POINT_BOX_UPPER_BOUND = 5
IN_FEATURES_LOWER_BOUND = 1
IN_FEATURES_UPPER_BOUND = 10000
MIN_VALUE_LOWER_BOUND = 0.01
MIN_VALUE_UPPER_BOUND = 10

class ConvexFunction(Function):
    """
    Represents a randomly generated convex n-dim funtion in form of: 
        f(x) = x.T * A * x + b.T * x + c

        Where:
            A - positive-definite symmetric matrix
            b - real vector
            c - real number
            x - real input vector

    That is generated in the following way:
        f(x) = s * (x - x0).T * H * (x - x0) + f0

        Where:
            s - real scale parameter
            H - normalized positive-definite symmetric matrix,
            x0 - function minimum point
            f0 - function minimum 

        Thus we get a representation of variables:
            A = s * H
            b = -2s * H * x0
            c = s * x0.T * H * x0 + f0

    Parameters:
        in_features (int): dimension of the input vector
        seed (int) : function generation seed
        scale (int) : function scale parameter 
    """

    def __init__ (self, in_features : Optional[int] = None, 
                        scale : Optional[float] = None,
                        seed : Optional[int] = None):
        self._seed = seed

        if in_features is not None and (in_features < IN_FEATURES_LOWER_BOUND or in_features > IN_FEATURES_UPPER_BOUND):
             raise ValueError(f"The specified parameter (in_features) is out of bounds [{IN_FEATURES_LOWER_BOUND}, {IN_FEATURES_UPPER_BOUND}].")

        if scale is not None and (scale < SCALE_LOWER_BOUND or scale > SCALE_UPPER_BOUND):
            raise ValueError(f"The specified parameter (scale) is out of bounds [{SCALE_LOWER_BOUND}, {SCALE_UPPER_BOUND}].")
        
        self._rng = get_rng(seed=seed, location_name="convex_function_class")

        if in_features is None:
            in_features = self._rng.integers(IN_FEATURES_LOWER_BOUND, IN_FEATURES_UPPER_BOUND)

        super().__init__(in_features=in_features)

        if scale is None:
            scale = self._rng.uniform(SCALE_LOWER_BOUND, SCALE_UPPER_BOUND)
        
        self._s = scale

        self._update_variables()
             
    def _update_variables(self):
        H = self._rng.uniform(size=(self.in_features, self.in_features))
        H = H.T @ H + 0.01 * np.eye(self.in_features)
        H = H / np.linalg.norm(H, 2)

        self._x0 = self._rng.uniform(OPT_POINT_BOX_LOWER_BOUND, OPT_POINT_BOX_UPPER_BOUND, size=self.in_features)
        self._f0 = self._rng.uniform(MIN_VALUE_LOWER_BOUND, MIN_VALUE_UPPER_BOUND)

        self._A = self._s * H
        self._b = -2 * self._s * H @ self._x0
        self._c = self._s * (self._x0 @ H @ self._x0) + self._f0
        
    
    def __call__(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Input vector dim is incorrect.")
        return float(x.T @ self._A @ x + self._b @ x + self._c)
    
    def get_gradient(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Input vector dim is incorrect.")
        return 2 * self._A @ x + self._b
    
    @property
    def A(self):
        return self._A.copy()
    
    @property
    def b(self):
        return self._b.copy()
    
    @property
    def c(self):
        return self._c
    
    @property
    def x0(self):
        return self._x0
    
    @property
    def f0(self):
        return self._f0
    
    @property
    def scale(self):
        return self._s
    
    @property
    def seed(self):
        return self._seed
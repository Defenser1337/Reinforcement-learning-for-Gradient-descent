import numpy as np
from typing import Optional

class ConvexFunction:
    """
    Represents a convex funtion in form of: 
        f(x) = x.T * A * x + b.T * x + c

    Where:
        A : positive-definite symmetric matrix of size N x N
        b : real vector of size N
        c : real number
        x : real vector variable of size N


    Parameters:
        in_features (int): dimension of the input vector
    """

    def __init__ (self, in_features : int, 
                  A : Optional[np.ndarray] = None, 
                  b : Optional[np.ndarray] = None, 
                  c : Optional[float] = None,
                  random_state : Optional[int] = None,
                  max_absolute_value : float = 1.0,
                  tol: float = 1e-10) -> None:
        self._in_features = in_features
        self._tol = tol
        self._max_absolute_value = max_absolute_value
        self._eps = 1e-12

        if random_state is not None:
            if A is not None or b is not None or c is not None:
                raise ValueError("Cannot use random generation and attribute definition at the same time.")
            
            rng = np.random.RandomState(random_state)

            # Let f(x) = (x_opt - x).T * A * (x_opt - x) + eps
            # Now we store x_opt value so that the function is positive

            self._x_opt = self.max_absolute_value * rng.uniform(-1, 1, size=self.in_features)
            
            self._A = ConvexFunction.generate_matrix(self.in_features, rng, self.max_absolute_value)
            self._b = -2 * self._x_opt.T @ self._A
            self._c = float(self._x_opt.T @ self._A @ self._x_opt + self.eps + self.max_absolute_value * rng.uniform(0, 1))
        elif A is not None and b is not None and c is not None:
            if max_absolute_value != 1.0:
                raise ValueError("Cannot use random generation and attribute definition at the same time")
            
            if A.ndim != 2 or A.shape[0] != self.in_features or A.shape[1] != self.in_features:
                raise ValueError("Matrix dimensions is incorrect")
            
            if not np.iscomplexobj(A):
                A = A.astype(np.complex128, copy=False)

            is_symmetric = np.allclose(A, A.conj().T, rtol=self.tol, atol=self.tol)
            is_positive_definite = np.all(np.linalg.eigvalsh(A) > self.tol)

            if not is_symmetric or not is_positive_definite:
                raise ValueError("Matrix should be positive-definite and symmetric")
            
            if b.ndim != 1 or b.shape[0] != self.in_features:
                raise ValueError("Vector dimensions is incorrect")
            
            if not isinstance(c, (int, float, np.number)):
                raise ValueError("Last attribute must be a scalar number")

            self.max_absolute_value = np.linalg.norm(self.A, 'fro')
            self._A = A
            self._b = b
            self._c = c
        else:
            raise ValueError("Constructor attributes is incorrect.")

    @staticmethod
    def generate_matrix(in_features: int, rng, max_absolute_value : float):
        """
        Generates positive-definite Hermitian matrix of size N x N
        
        Parameters 
            in_features (int): dimension of generated matrix
        """
        
        T = rng.uniform(-1, 1, size=(in_features, in_features))

        A = T @ T.conj().T + np.eye(in_features)

        return (A / np.linalg.norm(A, 'fro')) * max_absolute_value * in_features
    
    def __call__(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Vector dimension is incorrect")
        return float(x.T @ self._A @ x + self._b @ x + self._c)
    
    def get_gradient(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Vector dimension is incorrect")
        return 2 * self._A @ x + self._b
    
    @property
    def in_features(self):
        return self._in_features
    
    @property
    def max_absolute_value(self):
        return self._max_absolute_value
    
    @property
    def tol(self):
        return self._tol
    
    @property
    def eps(self):
        return self._eps
    
    @property
    def A(self):
        return self._A.copy()
    
    @property
    def b(self):
        return self._b.copy()
    
    @property
    def c(self):
        return self._c
import numpy as np
from typing import Optional, Union, Any

class ConvexFunction:
    """
    Represents a convex funtion in form of: 
        f(x) = x.T * A * x + b.T * x + c

    Where:
        A : positive-definite Hermitian matrix of size N x N
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
                  random_seed : Optional[int] = None,
                  max_absolute_value : float = 100.0,
                  tol: float = 1e-10) -> None:
        
        self.in_features = in_features
        self.tol = tol

        if random_seed is not None:
            if A is not None or b is not None or c is not None:
                raise ValueError("Cannot use random generation and attribute definition at the same time.")

            self.max_absolute_value = max_absolute_value

            rng = np.random.RandomState(random_seed)

            self._A = ConvexFunction.generate_matrix(self.in_features, rng, self.max_absolute_value)
            self._b = self.max_absolute_value * (2 * rng.rand(self.in_features) - 1)
            self._c = float(self.max_absolute_value * (2 * rng.rand() - 1))
        elif A is not None and b is not None and c is not None:
            if max_absolute_value != 1000.0:
                raise ValueError("Cannot use random generation and attribute definition at the same time.")
            
            if A.ndim != 2 or A.shape[0] != self.in_features or A.shape[1] != self.in_features:
                raise ValueError("Matrix dimensions is incorrect.")
            
            if not np.iscomplexobj(A):
                A = A.astype(np.complex128, copy=False)

            is_hermitian = np.allclose(A, A.conj().T, rtol=self.tol, atol=self.tol)
            is_positive_definite = np.all(np.linalg.eigvalsh(A) > self.tol)

            if not is_hermitian or not is_positive_definite:
                raise ValueError("Matrix should be positive-definite and Hermitian")
            
            if b.ndim != 1 or b.shape[0] != self.in_features:
                raise ValueError("Vector dimensions is incorrect.")
            
            if not isinstance(c, (int, float, np.number)):
                raise ValueError("Last attribute must be a scalar number")

            self._A = A
            self._b = b
            self._c = c

    @staticmethod
    def generate_matrix(in_features: int, rng, max_absolute_value):
        """
        Generates positive-definite Hermitian matrix of size N x N
        
        Parameters 
            in_features (int): dimension of generated matrix
        """
        
        #T = (rng.rand(in_features, in_features) + 1j * rng.rand(in_features, in_features))
        T = 2 * rng.rand(in_features, in_features) - 1

        A = T @ T.conj().T + np.eye(in_features)

        return (A / np.linalg.norm(A, 'fro')) * max_absolute_value * in_features
    
    def __call__(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Vector dimension is incorrect.")
        
        return float(x.T @ self._A @ x + self._b @ x + self._c)
    
    def get_gradient(self, x : np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.in_features:
                raise ValueError("Vector dimension is incorrect.")
        
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

    


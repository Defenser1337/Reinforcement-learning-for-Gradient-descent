import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from src.optimization.convex_function import ConvexFunction

class ConvexOptimization(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, in_features : int = 1, max_absolute_value : np.float32 = 100.0):
        # The count of function in_features 
        self.in_features = in_features
        self.tol = 0.001

        # The approximate maximum absoulute value of function coefs
        self.max_absolute_value = max_absolute_value

        # Using x = (0,0,...,0) as "uninitialized" state
        self._x = np.zeros(shape=self.in_features)

        self._function = None

        self.observation_space = spaces.Dict(
            {
                "grad_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32), 
                #"gradient_mean_EMA" : spaces.Box(low=1e-10, high=1e10, shape=(1,), dtype=np.float32),    
                #"gradient_var_EMA" : spaces.Box(low=1e-10, high=1e10, shape=(1,), dtype=np.float32),       
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options : Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            obj_seed = seed
        else:
            obj_seed = int(self.np_random.integers(0, 2**31 - 1))

        self._iteration = 0

        self._function = ConvexFunction(in_features=self.in_features, 
                                        random_state=obj_seed, 
                                        max_absolute_value=self.max_absolute_value)
        
        self._x = self.np_random.uniform(low = -self.max_absolute_value, high=self.max_absolute_value, size=(self.in_features,))
        self._grad = self._function.get_gradient(self._x)
        self._function_value = self._function(self._x)
        
        norm_val = np.linalg.norm(self._grad)
        self._grad_norm = np.array([np.clip(norm_val, 0, 1e10)], dtype=np.float32)

        grad_delta_norm = -np.inf
        
        observation = {
            "grad_norm": self._grad_norm
        }

        info = {
            "iteration" :   self._iteration,
            "function_value" : self._function_value,
            "grad_norm" : self._grad_norm, 
            "grad_delta_norm" : grad_delta_norm
        }

        return observation, info

    def step(self, action : np.ndarray):
        lr = 10**(action[0] * 5 - 2)

        self._iteration += 1

        prev_x = self._x.copy()
        prev_grad = self._grad.copy()
        prev_function_value = self._function_value
        prev_grad_norm = self._grad_norm

        self._x = prev_x - lr * prev_grad
        self._grad = self._function.get_gradient(self._x)
        self._function_value = self._function(self._x)

        norm_val = np.linalg.norm(self._grad)
        self._grad_norm = np.array([np.clip(norm_val, 0, 1e10)], dtype=np.float32)

        grad_delta_norm = np.linalg.norm(self._grad - prev_grad)

        truncated = bool(self._iteration > 5000)
        terminated = bool((self._grad_norm / (prev_grad_norm + 1e-12) > 1e6) 
                          or (self._grad_norm < self.tol)
                          or (not np.isfinite(self._grad_norm[0])))

        reward = np.log2(prev_function_value / (self._function_value + 1e-12))

        observation = {
            "grad_norm": self._grad_norm
        }

        info = {
            "iteration" :   self._iteration,
            "function_value" : self._function_value,
            "grad_norm" : self._grad_norm, 
            "grad_delta_norm" : grad_delta_norm
        }

        return observation, reward, terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from src.optimization.convex_function import ConvexFunction

class ConvexOptimization(gym.Env):
    """
    Gymnasium custom environment: 
    https://gymnasium.farama.org/introduction/create_custom_env/

    Represents convex optimization task procces where:
        Observation values:
            grad_norm: Gradient L2 norm,
            ...
            _TODO MORE_
            ...
        Action value:
            lr: normalized learning rate in logarithmic scale in the bound [-1, 1]
    Parameters:
        in_features (int): dimension of optimization task
        render_mode: Only accept "ansi" and None type
        max_absolute_value (float): set a box approximately where the minimum value is located
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, in_features : int = 1, render_mode = None, max_absolute_value : np.float32 = 1.0):
        self.render_mode = render_mode

        # The count of function in_features 
        self.in_features = in_features
        self.tol = 0.001

        # Set a box approximately where the minimum value is located
        self.max_absolute_value = max_absolute_value

        # Using x = (0,0,...,0) as "uninitialized" state
        self._x = np.zeros(shape=self.in_features)

        # Convex positive-semidefinite function f(x) for optimization 
        self._function = None

        self.observation_space = spaces.Dict({
            "grad_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)  
        })

        # We are choosing normalized learning rate in logarithmic scale
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.norm_clip = self.max_absolute_value**2

    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)

        if seed is not None:
            obj_seed = seed
        else:
            obj_seed = int(self.np_random.integers(low=0, high=2**31 - 1))

        self._iteration = 0

        self._function = ConvexFunction(in_features=self.in_features, 
                                        random_state=obj_seed, 
                                        max_absolute_value=self.max_absolute_value)
        
        self._x = self.np_random.uniform(low = -self.max_absolute_value, high=self.max_absolute_value, size=(self.in_features,))
        self._x0 = self._x.copy()
        self._grad = self._function.get_gradient(self._x)
        self._function_value = self._function(self._x)
        self._grad_delta_norm = -np.inf

        norm_val = np.linalg.norm(self._grad)
        self._grad_norm = np.array([np.clip(norm_val, 0, self.norm_clip)], dtype=np.float32)
        
        observation = {
            "grad_norm": self._grad_norm
        }

        info = {
            "iteration" :   self._iteration,
            "function_value" : self._function_value,
            "grad_norm" : self._grad_norm, 
            "grad_delta_norm" : self._grad_delta_norm,
            "x" : self._x
        }

        return observation, info

    def step(self, action : np.ndarray):
        self._iteration += 1

        lr = 10**(action[0] * 3 - 2)

        prev_x = self._x.copy()
        prev_grad = self._grad.copy()
        prev_function_value = self._function_value
        prev_grad_norm = self._grad_norm.copy()

        self._x = prev_x - lr * prev_grad
        self._grad = self._function.get_gradient(self._x)
        self._function_value = self._function(self._x)
        self._grad_delta_norm = np.linalg.norm(self._grad - prev_grad)

        norm_val = np.linalg.norm(self._grad)
        self._grad_norm = np.array([np.clip(norm_val, 0, self.norm_clip)], dtype=np.float32)

        eps = 1e-12

        reward = float(np.log2(prev_function_value + eps) - np.log2(self._function_value + eps))
        reward = np.clip(reward, -5.0, 5.0)

        observation = {
            "grad_norm": self._grad_norm
        }

        info = {
            "iteration" : self._iteration,
            "function_value" : self._function_value,
            "grad_norm" : self._grad_norm, 
            "grad_delta_norm" : self._grad_delta_norm,
            "x" : self._x,
            "status" : ""
        }

        diverged = bool(not(np.isfinite(self._grad).all() and np.isfinite(self._function_value)))
        converged = bool(self._grad_norm[0] < self.tol)

        truncated = bool(self._iteration > 10000)
        
        if diverged or ((self._grad_norm[0] / (prev_grad_norm[0] + eps)) > 1e3):
            reward += -100
            terminated = True
            info["status"] = "diverged"
        elif converged:
            reward += 100
            terminated = True
            info["status"] = "converged"
        else:
            terminated = False
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "ansi":
            render_string = (
                f"\n--- Iteration {self._iteration} ---\n"
                f"Function Value: {self._function_value:.6f}\n"
                f"Gradient Norm:  {np.linalg.norm(self._grad):.6f}\n"
                f"A: {self._function.A}\n"
                f"b: {self._function.b}\n"
                f"c: {self._function.c}\n"
                f"X start: {self._x0}\n"
                f"X best: {self._x}\n"
            )
            return render_string

    def close(self):
        pass

    def get_x_start(self):
        return self._x0.copy()
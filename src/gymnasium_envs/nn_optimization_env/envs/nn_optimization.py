import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

class NeuralNetworkOptimization(gym.Env):
    """
    Gymnasium custom environment: 
    https://gymnasium.farama.org/introduction/create_custom_env/

    Represents neural network optimization task process where:
        Observation values:
            l1_grad_norm: Gradient L1 norm ||∇F(Xt)||₂,
            l2_grad_norm: Gradient L2 norm ||∇F(Xt)||₁,
            l2_grad_norm_log: Logarithm of gradient L2 norm log(1 + ||∇F(Xt)||₂)
            cos_sim: Cosine similarity between gradient Cos(||∇F(Xt)||₂, ||∇F(Xt-1)||₂),
            grad_delta_norm: Norm of difference between gradient ||∇F(Xt)-∇F(Xt-1)||₂,
            function_value_delta_log: Logarithm of difference between function value SIGN * log(1 + f(Xt) - f(Xt-1))
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

    def __init__(self, render_mode = None):
        self.render_mode = render_mode

        self.tol = 0.001
        self.max_iterations = 10000

        self.observation_space = spaces.Dict({
            "l1_grad_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),  
            "l2_grad_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "l2_grad_norm_log" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),  
            "cos_sim" : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
            "grad_delta_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "function_value_delta_log" : spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        })

        # We are choosing normalized learning rate in logarithmic scale
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)


        observation = {
            "l1_grad_norm": np.array([l1_grad_norm_val], dtype=np.float32),
            "l2_grad_norm": np.array([l2_grad_norm_val], dtype=np.float32),
            "l2_grad_norm_log" : np.array([np.log1p(l2_grad_norm_val)], dtype=np.float32),
            "cos_sim" : np.array([0.0], dtype=np.float32),
            "grad_delta_norm": np.array([0.0], dtype=np.float32),
            "function_value_delta_log": np.array([0.0], dtype=np.float32),
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

        observation = {
            "l1_grad_norm": np.array([l1_grad_norm], dtype=np.float32),
            "l2_grad_norm": np.array([l2_grad_norm], dtype=np.float32),
            "l2_grad_norm_log" : np.array([l2_grad_norm_log], dtype=np.float32),
            "cos_sim" :  np.array([cos_sim], dtype=np.float32),
            "grad_delta_norm" : np.array([grad_delta_norm], dtype=np.float32),
            "function_value_delta_log" : np.array([function_value_delta_log], dtype=np.float32),
        }

        info = {
            "iteration" : self._iteration,
            "function_value" : self._function_value,
            "grad_norm" : l2_grad_norm, 
            "grad_delta_norm" : self._grad_delta_norm,
            "x" : self._x,
            "status" : ""
        }

        self._grad_norm = l2_grad_norm

        is_non_finite = (
            not np.isfinite(self._grad).all()
            or not np.isfinite(self._function_value)
        )
        is_exploding = (l2_grad_norm_curr / (l2_grad_norm_prev + eps)) > 1e3

        diverged   = bool(is_non_finite or is_exploding)
        converged  = bool(self._grad_norm < self.tol)
        truncated  = bool(self._iteration >= self.max_iterations)

        if diverged:
            reward += -100.0
            terminated = True
            info["status"] = "diverged"
        elif converged:
            reward += 100.0
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
    
    def get_function(self):
        return self._function
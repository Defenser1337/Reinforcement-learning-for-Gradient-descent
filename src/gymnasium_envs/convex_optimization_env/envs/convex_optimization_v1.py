import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from src.optimization.optimization_functions.convex_function import ConvexFunction
from src.optimization.optimization_functions.convex_function_w_noise import ConvexFunctionWithNoise

EPS = 1e-8
ALPHA = 1e-3

class ConvexOptimizationV1(gym.Env):
    """
    Represents convex function optimization task process.

    Gymnasium custom environment: 
    https://gymnasium.farama.org/introduction/create_custom_env/

    Version 1:  
        Reworked observation values:
            Deleted some observation values and added EMA normalizaion. 

    Observation values:
        grad_norm_scaled_log: 
            Logarithm of normalized gradient L2 norm log(1 + ||∇F||₂ / (EMA(||∇F||₂) + ϵ)),
        grad_delta_norm_scaled_log: 
            Logarithm of normalized norm of difference between gradient log(1 + ||∇F(Xt)-∇F(Xt-1)||₂ / (EMA(||∇F(Xt)-∇F(Xt-1)||₂) + ϵ)),
        cos_sim: 
            Cosine similarity between gradients Cos_sim(∇F(Xt), ∇F(Xt-1)),
        loss_scaled_log:
            Logarithm of normalized loss log(1 + F(X) / (EMA(F) + ϵ))
        loss_delta_scaled_log:
            Logarithm of normalized difference between losses sign(ΔF) * log(1 + |ΔF| / (|EMA(ΔF)| + ϵ))
        prev_action:
            Previous action

    Action value:
        lr: normalized learning rate in logarithmic scale in the bound [-1, 1]
        
    Parameters:
        in_features (int): dimension of optimization task
        render_mode: Only accept "ansi" and None type
        max_absolute_value (float): set a box approximately where the minimum value is located
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, in_features : int = 1, render_mode = None, max_absolute_value : np.float32 = 1.0, add_noise = False):
        self.render_mode = render_mode

        # The count of function in_features 
        self._in_features = in_features
        self._tol = 0.001
        self._max_iterations = 10000
        self._add_noise = add_noise

        # Set a box approximately where the minimum value is located
        self._max_absolute_value = max_absolute_value

        # Convex positive-semidefinite function f(x) for optimization 
        self._function = None

        self.observation_space = spaces.Dict({
            "grad_norm_scaled_log" : spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),  
            "grad_delta_norm_scaled_log" : spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "cos_sim" : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
            "loss_scaled_log" : spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "loss_delta_scaled_log" : spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
            "prev_action" : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # We are choosing learning rate in logarithmic scale
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)

        if seed is not None:
            obj_seed = seed
        else:
            obj_seed = int(self.np_random.integers(low=0, high=2**31 - 1))

        self._iteration = 0

        if self._add_noise == True:
            self._function = ConvexFunctionWithNoise(in_features=self._in_features, 
                                        random_state=obj_seed, 
                                        max_absolute_value=self._max_absolute_value)
        else:
            self._function = ConvexFunction(in_features=self._in_features, 
                                        random_state=obj_seed, 
                                        max_absolute_value=self._max_absolute_value)
        
        self._init_values()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action : np.ndarray):
        self._iteration += 1
        
        lr = 10**(action[0] * 3 - 2)

        self._prev_action = action[0]

        prev_x = self._curr_x.copy()
        self._curr_x = prev_x - lr * self._curr_grad

        self._update_values()

        observation = self._get_obs()
        info = self._get_info()

        reward = float(np.log2(self._prev_loss + EPS) - np.log2(self._curr_loss + EPS))
        reward = np.clip(reward, -5.0, 5.0)


        diverged   = bool(self._is_exploding)
        converged  = bool(self._curr_grad_norm < self._tol)
        truncated  = bool(self._iteration >= self._max_iterations)

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
                f"Function Value: {self._curr_loss:.6f}\n"
                f"Gradient Norm:  {self._curr_grad_norm:.6f}\n"
                f"A: {self._function.A}\n"
                f"b: {self._function.b}\n"
                f"c: {self._function.c}\n"
                f"X start: {self._start_x}\n"
                f"X best: {self._curr_x}\n"
            )
            return render_string

    def close(self):
        pass
    
    def get_x_start(self):
        return self._start_x.copy()
    
    def get_function(self):
        return self._function

    def _get_obs(self):
        grad_norm_scaled_log = np.log1p(
            self._curr_grad_norm / (self._ema_grad_norm + EPS)
        )

        grad_delta_norm_scaled_log = np.log1p(
            self._curr_grad_delta_norm / (self._ema_grad_delta_norm + EPS)
        )

        cos_sim = self._calculate_cos_sim()

        loss_scaled_log = np.log1p(
            self._curr_loss / (self._ema_loss + EPS)
        )

        abs_delta = abs(self._curr_loss_delta)
        sign = np.sign(self._curr_loss_delta)
        loss_delta_scaled_log = sign * np.log1p(
            abs_delta / (self._ema_loss_delta + EPS)
        )

        self._obs = {
            "grad_norm_scaled_log"      : np.array([grad_norm_scaled_log],       dtype=np.float32),
            "grad_delta_norm_scaled_log": np.array([grad_delta_norm_scaled_log], dtype=np.float32),
            "cos_sim"                   : np.array([cos_sim],                    dtype=np.float32),
            "loss_scaled_log"           : np.array([loss_scaled_log],            dtype=np.float32),
            "loss_delta_scaled_log"     : np.array([loss_delta_scaled_log],      dtype=np.float32),
            "prev_action"               : np.array([self._prev_action],          dtype=np.float32),
        }

        return self._obs

    def _get_info(self):
        return {
            "iteration"       : self._iteration,
            "loss"            : self._curr_loss,
            "grad_norm"       : self._curr_grad_norm, 
            "grad_delta_norm" : self._curr_grad_delta_norm,
            "x"               : self._curr_x,
            "status"          : "",
        }

    def _init_values(self):
        self._curr_x = self.np_random.uniform(low = -self._max_absolute_value, high=self._max_absolute_value, size=(self._in_features,))
        self._start_x = self._curr_x.copy()
        self._curr_grad = self._function.get_gradient(self._curr_x )
        self._prev_grad = None

        self._grad_norm_clip = None

        self._curr_grad_norm = np.linalg.norm(self._curr_grad)
        self._ema_grad_norm = self._curr_grad_norm
        self._prev_grad_norm = None

        self._curr_grad_delta_norm = 0.0
        self._ema_grad_delta_norm = EPS
        self._prev_grad_delta_norm = None

        self._curr_loss = self._function(self._curr_x )
        self._ema_loss = self._curr_loss
        self._prev_loss = None

        self._curr_loss_delta = 0.0
        self._ema_loss_delta = EPS
        self._prev_loss_delta = None

        self._prev_action = 0.0

    def _update_values(self):
        # Updating gradients
        self._prev_grad = self._curr_grad.copy()
        self._curr_grad = self._function.get_gradient(self._curr_x)

        # ---- GRAD NORM ----
        curr_grad_norm = np.linalg.norm(self._curr_grad)
        curr_grad_norm_isfinite = np.isfinite(curr_grad_norm)
        
        if curr_grad_norm_isfinite and self._grad_norm_clip is None:
            self._grad_norm_clip = max(1e6 * curr_grad_norm, 1.0)

        if not curr_grad_norm_isfinite:
            curr_grad_norm = self._grad_norm_clip if self._grad_norm_clip is not None else 0.0
        else:
            if self._grad_norm_clip is not None:
                curr_grad_norm = min(curr_grad_norm, self._grad_norm_clip)

        self._ema_grad_norm = ALPHA * self._curr_grad_norm + (1 - ALPHA) * self._ema_grad_norm
        self._prev_grad_norm = self._curr_grad_norm
        self._curr_grad_norm = curr_grad_norm

        # ---- GRAD DELTA NORM----
        if self._prev_grad is not None:
            prev_valid = np.all(np.isfinite(self._prev_grad))
            curr_valid = curr_grad_norm_isfinite

            if prev_valid and curr_valid:
                grad_delta_norm = np.linalg.norm(self._curr_grad - self._prev_grad)
                if not np.isfinite(grad_delta_norm):
                    grad_delta_norm = self._grad_norm_clip if self._grad_norm_clip is not None else 0.0
                elif self._grad_norm_clip is not None:
                    grad_delta_norm = min(grad_delta_norm, self._grad_norm_clip)
            else:
                grad_delta_norm = 0.0
        else:
            grad_delta_norm = 0.0


        self._ema_grad_delta_norm = ALPHA * self._curr_grad_delta_norm + (1 - ALPHA) * self._ema_grad_delta_norm
        self._prev_grad_delta_norm = self._curr_grad_delta_norm
        self._curr_grad_delta_norm = grad_delta_norm

        # ---- LOSS ----
        self._prev_loss = self._curr_loss
        self._ema_loss = ALPHA * self._curr_loss + (1 - ALPHA) * self._ema_loss
        self._curr_loss = self._function(self._curr_x)

        if not np.isfinite(self._curr_loss):
            self._curr_loss = self._prev_loss

        # ---- LOSS DELTA ----
        if self._prev_loss is not None:
            loss_delta = self._curr_loss - self._prev_loss
            if not np.isfinite(loss_delta):
                loss_delta = 0.0
        else:
            loss_delta = 0.0

        self._ema_loss_delta = ALPHA * abs(self._curr_loss_delta) + (1 - ALPHA) * self._ema_loss_delta
        self._prev_loss_delta = self._curr_loss_delta
        self._curr_loss_delta = loss_delta

        # ---- EXPLODING ----
        self._is_exploding = (
            np.isfinite(self._curr_grad_norm)
            and np.isfinite(self._prev_grad_norm)
            and self._prev_grad_norm > EPS
            and (self._curr_grad_norm / self._prev_grad_norm) > 1e3
        )

    def _calculate_cos_sim(self):
        if self._prev_grad is None or self._prev_grad.shape != self._curr_grad.shape:
            return 0.0

        if not np.all(np.isfinite(self._curr_grad)):
            return 0.0

        if not np.all(np.isfinite(self._prev_grad)):
            return 0.0

        if self._curr_grad_norm <= 1e-12 or self._prev_grad_norm <= 1e-12:
            return 0.0
        
        grad_unit = self._curr_grad / self._curr_grad_norm
        prev_grad_unit = self._prev_grad / self._prev_grad_norm
        
        cos_sim = np.dot(grad_unit, prev_grad_unit)
        
        if not np.isfinite(cos_sim):
            return 0.0
        
        return float(np.clip(cos_sim, -1.0, 1.0))
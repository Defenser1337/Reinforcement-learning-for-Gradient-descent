import torch
from torch.optim import Adam
import numpy as np
import math

EPS   = 1e-8
ALPHA = 0.01

class CustomLRV1(Adam):
    def __init__(self, params, lr=0.01, betas = (0.9, 0.999), learn_betas = False):
        super().__init__(params, lr=lr, betas=betas, eps=EPS)

        self._learn_betas = learn_betas

        # gradient state 
        self._prev_grad        = None
        self._curr_grad        = None
        self._curr_l2_norm     = None
        self._prev_l2_norm     = None
        self._l2_norm_clip     = None
        self._curr_grad_valid  = False

        # loss state 
        self._curr_loss        = None
        self._prev_loss        = None

        # EMA state 
        self._ema_grad_norm        = None   # initialised on first call
        self._ema_grad_delta_norm  = EPS
        self._ema_loss             = None   # initialised on first call
        self._ema_loss_delta       = EPS

        # delta state (needed for EMA update ordering)
        self._curr_grad_delta_norm = 0.0
        self._curr_loss_delta      = 0.0

        # misc 
        self._prev_lr = 0.0

        if learn_betas is True:
            self._prev_beta1 = 0.9
            self._prev_beta2 = 0.999

    @torch.no_grad()
    def step(self, closure=None, lr=None, beta1=None, beta2=None):
        if lr is not None:
            for group in self.param_groups:
                group['lr'] = lr
        
        if self._learn_betas and beta1 is not None and beta2 is not None:
            for group in self.param_groups:
                group['betas'] = (beta1, beta2)

        super().step(closure)
        

    @torch.no_grad()
    def get_obs(self, loss: float, prev_lr: float = 0.0, prev_beta1 : float = 0.9, prev_beta2 : float = 0.999):
        flat_grad = self._collect_flat_grad()

        if flat_grad is None:
            return self._zero_obs(prev_lr, prev_beta1, prev_beta2)

        self._update_state(flat_grad, loss)

        # Build observation
        grad_norm_scaled_log = float(np.log1p(
            self._curr_l2_norm / (self._ema_grad_norm + EPS)
        ))

        grad_delta_norm_scaled_log = float(np.log1p(
            self._curr_grad_delta_norm / (self._ema_grad_delta_norm + EPS)
        ))

        cos_sim = self._calculate_cos_sim()

        loss_scaled_log = float(np.log1p(
            self._curr_loss / (self._ema_loss + EPS)
        ))

        abs_delta = abs(self._curr_loss_delta)
        sign      = float(np.sign(self._curr_loss_delta))
        loss_delta_scaled_log = sign * float(np.log1p(
            abs_delta / (self._ema_loss_delta + EPS)
        ))

        obs = {
            "grad_norm_scaled_log":       np.array([grad_norm_scaled_log],       dtype=np.float32),
            "grad_delta_norm_scaled_log": np.array([grad_delta_norm_scaled_log], dtype=np.float32),
            "cos_sim":                    np.array([cos_sim],                    dtype=np.float32),
            "loss_scaled_log":            np.array([loss_scaled_log],            dtype=np.float32),
            "loss_delta_scaled_log":      np.array([loss_delta_scaled_log],      dtype=np.float32),
            "prev_lr":                    np.array([prev_lr],          dtype=np.float32),
        }

        if self._learn_betas is True:
            obs["prev_beta1"] = np.array([prev_beta1], dtype=np.float32)
            obs["prev_beta2"] = np.array([prev_beta2], dtype=np.float32)
        
        return obs



    @torch.no_grad()
    def get_info(self, iteration: int):
        return {
            "iteration":       iteration,
            "loss":            self._curr_loss  if self._curr_loss  is not None else 0.0,
            "grad_norm":       self._curr_l2_norm if self._curr_l2_norm is not None else 0.0,
            "grad_delta_norm": self._curr_grad_delta_norm,
            "status":          "",
        }

    @torch.no_grad()
    def is_diverged(self) -> bool:
        if self._curr_loss is None:
            return False
        if not math.isfinite(self._curr_loss):
            return True
        if not self._curr_grad_valid:
            return True
        return False


    # Internal state update
    def _update_state(self, flat_grad: torch.Tensor, loss: float):
        curr_l2_norm = flat_grad.norm(2).item()
        curr_l2_norm_isfinite = math.isfinite(curr_l2_norm)

        self._curr_grad_valid = curr_l2_norm_isfinite

        # Initialise clip on first valid observation
        if curr_l2_norm_isfinite and self._l2_norm_clip is None:
            self._l2_norm_clip = max(1e6 * curr_l2_norm, 1.0)

        # Apply clip 
        if not curr_l2_norm_isfinite:
            curr_l2_norm = self._l2_norm_clip if self._l2_norm_clip is not None else 0.0
        elif self._l2_norm_clip is not None:
            curr_l2_norm = min(curr_l2_norm, self._l2_norm_clip)

        # ---- GRAD NORM ----
        if self._ema_grad_norm is None:
            self._ema_grad_norm = curr_l2_norm      
        else:
            prev_norm = self._curr_l2_norm if self._curr_l2_norm is not None else curr_l2_norm
            self._ema_grad_norm = ALPHA * prev_norm + (1 - ALPHA) * self._ema_grad_norm

        self._prev_l2_norm = self._curr_l2_norm
        self._curr_l2_norm = curr_l2_norm

        # ---- GRAD DELTA NORM----

        if self._prev_grad is not None and self._curr_grad_valid:
            if not (self._prev_grad.isnan().any() or self._prev_grad.isinf().any()):
                delta_norm = (flat_grad - self._prev_grad).norm(2).item()
                if not math.isfinite(delta_norm):
                    delta_norm = self._l2_norm_clip if self._l2_norm_clip is not None else 0.0
                elif self._l2_norm_clip is not None:
                    delta_norm = min(delta_norm, self._l2_norm_clip)
            else:
                delta_norm = 0.0
        else:
            delta_norm = 0.0

        self._ema_grad_delta_norm  = ALPHA * self._curr_grad_delta_norm + (1 - ALPHA) * self._ema_grad_delta_norm
        self._curr_grad_delta_norm = delta_norm

        # ---- LOSS ----
        if self._ema_loss is None:
            self._ema_loss = loss   
        else:
            self._ema_loss = ALPHA * loss + (1 - ALPHA) * self._ema_loss

        loss_delta = (loss - self._curr_loss) if self._curr_loss is not None else 0.0
        if not math.isfinite(loss_delta):
            loss_delta = 0.0

        self._ema_loss_delta  = ALPHA * abs(self._curr_loss_delta) + (1 - ALPHA) * self._ema_loss_delta
        self._curr_loss_delta = loss_delta

        self._prev_loss = self._curr_loss
        self._curr_loss = loss

        # Store grad for next step
        if self._curr_grad_valid:
            self._prev_grad = flat_grad.clone()
        self._curr_grad = flat_grad

    def _collect_flat_grad(self):
        grads = [
            p.grad.flatten()
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.cat(grads) if grads else None

    def _calculate_cos_sim(self) -> float:
        if self._prev_grad is None or self._prev_grad.shape != self._curr_grad.shape:
            return 0.0
        if not self._curr_grad_valid:
            return 0.0
        if self._prev_grad.isnan().any() or self._prev_grad.isinf().any():
            return 0.0

        if self._prev_l2_norm is None or self._curr_l2_norm is None:
            return 0.0
        if self._curr_l2_norm <= 1e-12 or self._prev_l2_norm <= 1e-12:
            return 0.0

        cos_sim = torch.dot(
            self._curr_grad / self._curr_l2_norm,
            self._prev_grad / self._prev_l2_norm,
        ).item()

        return float(np.clip(cos_sim, -1.0, 1.0)) if math.isfinite(cos_sim) else 0.0

    def _zero_obs(self, prev_lr: float, prev_beta1 : float, prev_beta2 : float) -> dict:
        zero = np.array([0.0], dtype=np.float32)

        obs = {
            "grad_norm_scaled_log":       zero.copy(),
            "grad_delta_norm_scaled_log": zero.copy(),
            "cos_sim":                    zero.copy(),
            "loss_scaled_log":            zero.copy(),
            "loss_delta_scaled_log":      zero.copy(),
            "prev_lr":                    np.array([prev_lr], dtype=np.float32)
        }
        
        if self._learn_betas is True:
            obs["prev_beta1"] = np.array([prev_beta1], dtype=np.float32)
            obs["prev_beta2"] = np.array([prev_beta2], dtype=np.float32)

        return obs

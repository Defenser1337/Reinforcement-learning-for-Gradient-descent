import torch
from torch.optim import Optimizer
import numpy as np
import math

class CustomLR(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
        self._prev_grad = None
        self._prev_loss = None
        self._curr_grad = None
        self._curr_l1_norm = None
        self._curr_l2_norm = None
        self._l1_norm_clip = None
        self._l2_norm_clip = None
        self._curr_grad_valid = False
        self._prev_l2_norm = None        
        self._curr_loss = None           
        self._curr_l1_norm_isfinite = False 
        self._curr_l2_norm_isfinite = False
    
    @torch.no_grad()
    def step(self, closure=None, lr=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            effective_lr = lr if lr is not None else group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                p.add_(p.grad, alpha=-effective_lr)

        return 
    @torch.no_grad()
    def get_obs(self, loss : float):
        flat_grad = self._collect_flat_grad()

        if flat_grad is None:
            return None

        self._update_grad_params(flat_grad, loss)

        l1_grad_norm = self._calculate_l1_grad_norm()
        l2_grad_norm = self._calculate_l2_grad_norm()
        l2_grad_norm_log = np.log1p(l2_grad_norm)
        cos_sim = self._calculate_cos_sim()
        grad_delta_norm = self._calculate_grad_delta_norm()
        loss_delta_log = self._calculate_loss_delta_log()
        loss_log = self._calculate_loss_log()

        self._update_prev_grad_params()

        observation = {
            "l1_grad_norm": np.array([l1_grad_norm], dtype=np.float32),
            "l2_grad_norm": np.array([l2_grad_norm], dtype=np.float32),
            "l2_grad_norm_log" : np.array([l2_grad_norm_log], dtype=np.float32),
            "cos_sim" :  np.array([cos_sim], dtype=np.float32),
            "grad_delta_norm" : np.array([grad_delta_norm], dtype=np.float32),
            "loss_delta_log" : np.array([loss_delta_log], dtype=np.float32),
            "loss_log" : np.array([loss_log], dtype=np.float32),
        }

        return observation
    
    @torch.no_grad()
    def get_info(self, iteration):
        info = {
            "iteration": iteration,
            "loss": self._curr_loss   if self._curr_loss   is not None else 0.0,
            "grad_norm": self._curr_l2_norm if self._curr_l2_norm is not None else 0.0,
            "grad_delta_norm": self._calculate_grad_delta_norm(),
            "status": ""
        }

        return info
    
    @torch.no_grad()
    def is_diverged(self):
        if self._curr_loss is None:
            return False
        
        if not math.isfinite(self._curr_loss):
            return True
        
        if not self._curr_grad_valid:
            return True
        
        return False

    def _collect_flat_grad(self):
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.flatten())
        return torch.cat(grads) if grads else None

    def _update_grad_params(self, flat_grad, loss):
        self._curr_grad = flat_grad
        self._curr_l1_norm = flat_grad.norm(1).item()
        self._curr_l2_norm = flat_grad.norm(2).item()

        self._curr_l1_norm_isfinite = math.isfinite(self._curr_l1_norm)
        self._curr_l2_norm_isfinite = math.isfinite(self._curr_l2_norm)

        self._curr_grad_valid = self._curr_l1_norm_isfinite and self._curr_l2_norm_isfinite

        if self._curr_l1_norm_isfinite and self._l1_norm_clip is None:
            self._l1_norm_clip = max(1e6 * self._curr_l1_norm, 1.0)
        if self._curr_l2_norm_isfinite and self._l2_norm_clip is None:
            self._l2_norm_clip = max(1e6 * self._curr_l2_norm, 1.0)

        self._curr_loss = loss

    def _update_prev_grad_params(self):
        if self._curr_grad_valid:
            self._prev_grad    = self._curr_grad.clone()
            self._prev_l2_norm = self._curr_l2_norm
        self._prev_loss = self._curr_loss 

    def _calculate_l1_grad_norm(self):
        if not self._curr_l1_norm_isfinite and self._l1_norm_clip is None:
            return 0.0
        if not self._curr_l1_norm_isfinite:
            return self._l1_norm_clip
        return min(self._curr_l1_norm, self._l1_norm_clip)
    
    def _calculate_l2_grad_norm(self):
        if not self._curr_l2_norm_isfinite and self._l2_norm_clip is None:
            return 0.0
        if not self._curr_l2_norm_isfinite:
            return self._l2_norm_clip
        return min(self._curr_l2_norm, self._l2_norm_clip)
    
    def _calculate_cos_sim(self):
        if self._prev_grad is None or self._prev_grad.shape != self._curr_grad.shape:
            return 0.0
        if not self._curr_grad_valid:
            return 0.0
        if self._prev_grad.isnan().any() or self._prev_grad.isinf().any():
            return 0.0
        if self._curr_l2_norm <= 1e-12 or self._prev_l2_norm <= 1e-12:
            return 0.0

        grad_unit = self._curr_grad / self._curr_l2_norm
        prev_grad_unit = self._prev_grad / self._prev_l2_norm

        cos_sim = torch.dot(grad_unit, prev_grad_unit).item()

        if not math.isfinite(cos_sim):
            return 0.0
        return max(-1.0, min(1.0, cos_sim))
    
    def _calculate_grad_delta_norm(self):
        if self._prev_grad is None or self._prev_grad.shape != self._curr_grad.shape:
            return 0.0
        if not self._curr_grad_valid:
            return 0.0
        if self._prev_grad.isnan().any() or self._prev_grad.isinf().any():
            return 0.0

        grad_delta_norm = (self._curr_grad - self._prev_grad).norm(2).item()

        if not math.isfinite(grad_delta_norm):
            return 0.0
        
        if self._l2_norm_clip is None:
            return grad_delta_norm
        
        return min(grad_delta_norm, self._l2_norm_clip)
    
    def _calculate_loss_delta_log(self):
        if self._curr_loss is not None and self._prev_loss is not None:
            delta = self._curr_loss - self._prev_loss
            sign = 1.0 if delta >= 0 else -1.0
            function_value_delta_log = sign * math.log1p(abs(delta))
        else:
            function_value_delta_log = 0.0

        return function_value_delta_log
    
    def _calculate_loss_log(self):
        loss_log = np.log1p(self._curr_loss)

        return loss_log
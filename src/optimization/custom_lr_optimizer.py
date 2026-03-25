import torch
from torch.optim import Optimizer

class CustomLR(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
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

        return loss
"""
sam.py

Sharpness-Aware Minimization (SAM) Optimizer wrapper (Foret et al., ICLR 2021).
Tim cuc tieu phang (flat minima) bang cach toi uu dong thoi gia tri loss va do sac nhon
trong vung lan can ban kinh rho.
"""
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # leo len diem co loss cao nhat trong ban kinh rho: w + e(w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # quay ve w ban dau truoc khi cap nhat
        self.base_optimizer.step()               # cap nhat w bang gradient tai w + e(w)
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.detach().norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise NotImplementedError("SAM yeu cau goi first_step() va second_step().")


def disable_running_stats(model):
    """Tam dung cap nhat running_mean/var cua BatchNorm o buoc forward thu 2 cua SAM."""
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    """Khoi phuc cap nhat running_mean/var cua BatchNorm o buoc forward thu 1 cua SAM."""
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)

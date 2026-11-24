"""
sam.py

Sharpness-Aware Minimization (SAM) Optimizer wrapper (Foret et al., ICLR 2021).
Tim cuc tieu phang (flat minima) bang cach toi uu dong thoi gia tri loss va do sac nhon
trong vung lan can ban kinh rho.
"""
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        base_rho = kwargs.pop("base_rho", None)
        defaults = dict(rho=rho, sam_rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        if base_rho is not None:
            kwargs["rho"] = base_rho
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["sam_rho"] = rho
            group["adaptive"] = adaptive
            if base_rho is not None:
                group["rho"] = base_rho
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        has_sophia = hasattr(self.base_optimizer, "update_param_hessian")
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            sam_rho = group.get("sam_rho", group["rho"])
            scale = sam_rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if has_sophia:
                    self.state[p]["old_g"] = p.grad.detach().clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # leo len diem co loss cao nhat trong ban kinh sam_rho: w + e(w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, **step_kwargs):
        has_sophia = hasattr(self.base_optimizer, "update_param_hessian")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])  # copy in-place de giu contiguous memory cho cuDNN LSTM
                if has_sophia and "old_g" in self.state[p]:
                    self.base_optimizer.update_param_hessian(
                        p, p.grad.detach() - self.state[p]["old_g"]
                    )
        self.base_optimizer.step(**step_kwargs)       # cap nhat w bang gradient tai w + e(w)
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        vals = []
        for group in self.param_groups:
            if group["adaptive"]:
                for p in group["params"]:
                    if p.grad is not None:
                        vals.append((torch.abs(p) * p.grad).detach())
            else:
                for p in group["params"]:
                    if p.grad is not None:
                        vals.append(p.grad.detach())
        if not vals:
            return torch.tensor(0.0, device=shared_device)
        norms = [n.to(shared_device) for n in torch._foreach_norm(vals, 2)]
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

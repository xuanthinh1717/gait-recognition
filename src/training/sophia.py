"""
sophia.py

SophiaG: Second-order Clipped Stochastic Optimizer (Liu et al., ICLR 2024).
Su dung xap xi duong cheo ma tran Hessian (Gauss-Newton-Bartlett estimator) cap nhat dinh ky
moi k buoc va co che clipping theo do cong de khoan sau xuong day cuc tieu.
Co bias-correction cho ca exp_avg va hessian de hoi tu chinh xac ngay tu cac epoch dau.
"""
import torch
from torch.optim import Optimizer


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-4, eps=1e-12):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        """Cap nhat EMA cua duong cheo Hessian h_t = beta2 * h_{t-1} + (1 - beta2) * g_hat^2."""
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["hessian_step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                state["hessian_step"] += 1
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, bs=128):
        """
        Cap nhat tham so voi curvature-aware clipping:
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            w_{t+1} = w_t * (1 - lr * wd) - lr * clip(m_hat_t / max(rho * bs * h_hat_t, eps), -1, 1)
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            rho = group["rho"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["hessian_step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                bc1 = 1.0 - beta1 ** state["step"]
                m_hat = exp_avg / bc1

                if state["hessian_step"] > 0:
                    bc2 = 1.0 - beta2 ** state["hessian_step"]
                    h_hat = hessian / bc2
                else:
                    h_hat = grad * grad

                ratio = (m_hat / (rho * bs * h_hat).clamp(min=eps)).clamp(-1.0, 1.0)
                p.add_(ratio, alpha=-lr)

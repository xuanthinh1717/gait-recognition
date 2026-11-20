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
                 weight_decay=1e-4, eps=1e-12, ef_weight=0.0):
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
        if not 0.0 <= ef_weight <= 1.0:
            raise ValueError(f"Invalid ef_weight value: {ef_weight}")

        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, eps=eps, ef_weight=ef_weight)
        super().__init__(params, defaults)

    def _init_state(self, state, p):
        state["step"] = 0
        state["hessian_step"] = 0
        state["ef_step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["hessian"] = torch.zeros_like(p)
        state["ef_hessian"] = torch.zeros_like(p)

    @torch.no_grad()
    def update_hessian(self):
        """Cap nhat EMA cua duong cheo Hessian GNB: h_t = beta2 * h_{t-1} + (1 - beta2) * g_hat^2."""
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)
                state["hessian_step"] += 1
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def update_ef_hessian(self):
        """Cap nhat EMA do cong Empirical Fisher tu gradient cua toan bo ham loss (CE + Triplet)."""
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)
                state["ef_step"] += 1
                state["ef_hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, bs=128):
        """
        Cap nhat tham so voi curvature-aware clipping:
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            w_{t+1} = w_t * (1 - lr * wd) - lr * clip(m_hat_t / max(rho * bs * h_hat_t, eps), -1, 1)
        Neu ef_weight > 0: pha tron do cong GNB va do cong Empirical Fisher toan phan (CE + Triplet)
        de tranh h_hat suy bien ve 0 (SignSGD) khi train_acc -> 100%.
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            rho = group["rho"]
            eps = group["eps"]
            ef_w = group.get("ef_weight", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                bc1 = 1.0 - beta1 ** state["step"]
                m_hat = exp_avg / bc1

                if ef_w > 0.0:
                    state["ef_step"] += 1
                    state["ef_hessian"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bc_ef = 1.0 - beta2 ** state["ef_step"]
                    h_ef = state["ef_hessian"] / bc_ef
                else:
                    h_ef = None

                if state["hessian_step"] > 0:
                    bc2 = 1.0 - beta2 ** state["hessian_step"]
                    h_gnb = hessian / bc2
                else:
                    h_gnb = grad * grad

                if ef_w >= 1.0:
                    h_hat = h_ef
                elif ef_w > 0.0:
                    h_hat = (1.0 - ef_w) * h_gnb + ef_w * h_ef
                else:
                    h_hat = h_gnb

                ratio = (m_hat / (rho * bs * h_hat).clamp(min=eps)).clamp(-1.0, 1.0)
                p.add_(ratio, alpha=-lr)

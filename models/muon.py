import torch

# Standalone Muon Optimizer (Delegates 1D vectors to native AdamW, matrix weights to Newton-Schulz)
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, adamw_lr=1e-3, eps=1e-5):
        matrix_params = [p for p in params if p.ndim >= 2]
        vector_params = [p for p in params if p.ndim < 2]
        self.adamw = torch.optim.AdamW(vector_params, lr=adamw_lr, eps=eps) if vector_params else None
        super().__init__(matrix_params, dict(lr=lr, momentum=momentum, eps=eps))

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def newton_schulz(self, G, steps=5, eps=1e-5):
        # Newton-Schulz 5th-order polynomial matrix orthogonalization for Muon
        assert G.ndim == 2
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def apply_update_2d(self, p, buf, lr, eps=1e-5):
        scale = lr * max(1, p.shape[0] / p.shape[1]) ** 0.5
        update = self.newton_schulz(buf, eps=eps)
        p.sub_(update, alpha=scale)

    def apply_update_3d(self, p, buf, lr, eps=1e-5):
        for layer_idx in range(p.shape[0]):
            self.apply_update_2d(p[layer_idx], buf[layer_idx], lr, eps)

    def apply_update(self, p, buf, lr, eps=1e-5):
        match p.ndim:
            case 2:
                self.apply_update_2d(p, buf, lr, eps)
            case 3:
                self.apply_update_3d(p, buf, lr, eps)
            case _:
                raise ValueError(f"Muon optimizer only supports 2D or 3D parameters, got ndim={p.ndim}")

    def step_param(self, p, group):
        if p.grad is None:
            return

        lr, momentum, eps = group["lr"], group["momentum"], group["eps"]
        state = self.state[p]
        if "momentum_buf" not in state:
            state["momentum_buf"] = torch.zeros_like(p)

        buf = state["momentum_buf"]
        buf.mul_(momentum).add_(p.grad, alpha=1.0 - momentum)
        self.apply_update(p, buf, lr, eps)

    def step_group(self, group):
        for p in group["params"]:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.adamw:
            self.adamw.step()

        for group in self.param_groups:
            self.step_group(group)

        return loss

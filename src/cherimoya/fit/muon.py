"""Local implementation of the Muon optimizer.

Adapted from torch/optim/_muon.py (PyTorch 2.9+) with private PyTorch internals
removed so that it runs on older PyTorch versions. The algorithm and default
hyperparameters are identical to the upstream implementation.

Reference: https://kellerjordan.github.io/posts/muon/
"""

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["Muon"]

# Newton-Schulz polynomial coefficients from Keller Jordan's Muon post.
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Newton-Schulz orthogonalization of a 2D gradient matrix."""
    if ns_steps >= 100:
        raise ValueError("ns_steps must be less than 100")
    if grad.ndim != 2:
        raise ValueError("Input gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    ortho = grad.bfloat16()
    if ortho.size(0) > ortho.size(1):
        ortho = ortho.T

    ortho.div_(ortho.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        ortho = torch.addmm(
            torch.addmm(gram, gram, gram, beta=b, alpha=c),
            ortho,
            alpha=a,
        )

    if grad.size(0) > grad.size(1):
        ortho = ortho.T
    return ortho


def _adjust_lr(
    lr: float,
    adjust_lr_fn: str | None,
    param_shape: torch.Size,
) -> float:
    """Per-matrix learning rate scaling."""
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(A, B))
    return lr


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
) -> None:
    lr = float(lr)
    for i, param in enumerate(params):
        grad = grads[i]
        buf = momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


class Muon(Optimizer):
    """Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

    Applies Newton-Schulz orthogonalization to a Nesterov momentum buffer
    before each weight update. Designed for 2D weight matrices in hidden
    layers; use AdamW for all other parameters (biases, embeddings, etc.).

    Args:
        params: Iterable of 2D parameters to optimize.
        lr: Learning rate (default: 1e-3).
        weight_decay: Decoupled weight decay (default: 0.1).
        momentum: Momentum factor (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_coefficients: Newton-Schulz polynomial coefficients (a, b, c)
            (default: (3.4445, -4.7750, 2.0315)).
        eps: Numerical stability epsilon (default: 1e-7).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        adjust_lr_fn: LR scaling strategy — "original" (sqrt(max(1, A/B)))
            or "match_rms_adamw" (0.2*sqrt(max(A,B))). Default is "original".
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate must be >= 0, got {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Momentum must be >= 0, got {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Unknown adjust_lr_fn: {adjust_lr_fn!r}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters; got shape {tuple(p.shape)}"
                    )

    def _init_group(
        self,
        group: dict,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
    ) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            momentum_bufs.append(state["momentum_buffer"])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []
            self._init_group(group, params_with_grad, grads, momentum_bufs)

            _single_tensor_muon(
                params_with_grad,
                grads,
                momentum_bufs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                ns_steps=group["ns_steps"],
                eps=group["eps"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )

        return loss

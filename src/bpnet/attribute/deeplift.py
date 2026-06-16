"""DeepLIFT helpers that allow soft reference PFMs."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy
import torch
from bpnetlite.bpnet import _ProfileLogitScaling
from bpnetlite.chrombpnet import _Exp, _Log
from tangermeme.deep_lift_shap import (
    AttributionReferencesResult,
    _autocast_supported,
    _clear_hooks,
    _maxpool,
    _nonlinear,
    _register_hooks,
    _resolve_device,
    _softmax,
    hypothetical_attributions,
)
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import _validate_input
from tqdm import trange


def _deep_lift_shap_no_reference_ohe_check(
    model: torch.nn.Module,
    X: torch.Tensor,
    args: tuple | None = None,
    target: int = 0,
    batch_size: int = 32,
    references: Callable[..., Any] | torch.Tensor = dinucleotide_shuffle,
    n_shuffles: int = 20,
    return_references: bool = False,
    hypothetical: bool = False,
    warning_threshold: float = 0.001,
    additional_nonlinear_ops: dict | None = None,
    print_convergence_deltas: bool = False,
    raw_outputs: bool = False,
    only_warn: bool = False,
    dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    random_state: int | numpy.random.RandomState | None = None,
    verbose: bool = False,
) -> torch.Tensor | AttributionReferencesResult:
    """Tangermeme DeepLIFT/SHAP with tensor-reference OHE validation removed.

    ``X`` is still required to be one-hot encoded. Tensor references are only
    shape-checked so soft PFM references can be used as DeepLIFT baselines.
    """

    _validate_input(X, "X", shape=(-1, -1, -1), ohe=True, only_warn=only_warn)

    if X.shape[0] == 0:
        raise ValueError(
            "deep_lift_shap requires at least one example; got X with shape[0] == 0."
        )

    nonlinear_ops = {
        torch.nn.ReLU: _nonlinear,
        torch.nn.ReLU6: _nonlinear,
        torch.nn.RReLU: _nonlinear,
        torch.nn.SELU: _nonlinear,
        torch.nn.CELU: _nonlinear,
        torch.nn.GELU: _nonlinear,
        torch.nn.SiLU: _nonlinear,
        torch.nn.Mish: _nonlinear,
        torch.nn.GLU: _nonlinear,
        torch.nn.ELU: _nonlinear,
        torch.nn.LeakyReLU: _nonlinear,
        torch.nn.Sigmoid: _nonlinear,
        torch.nn.Tanh: _nonlinear,
        torch.nn.Softplus: _nonlinear,
        torch.nn.Softshrink: _nonlinear,
        torch.nn.LogSigmoid: _nonlinear,
        torch.nn.PReLU: _nonlinear,
        torch.nn.MaxPool1d: _maxpool,
        torch.nn.MaxPool2d: _maxpool,
        torch.nn.Softmax: _softmax,
    }

    device = _resolve_device(device)

    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if additional_nonlinear_ops is not None:
        for key, value in additional_nonlinear_ops.items():
            nonlinear_ops[key] = value

    use_autocast = _autocast_supported(device, dtype)

    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = None
    was_training = model.training

    model.to(device).eval()

    try:
        for module in model.modules():
            module._NON_LINEAR_OPS = nonlinear_ops

        try:
            model.apply(_register_hooks)
        except Exception as error:
            model.apply(_clear_hooks)
            raise error

        attributions, references_, Xi, rj, attr_ = [], [], [], [], []
        if isinstance(references, torch.Tensor):
            _validate_input(
                references,
                "references",
                shape=(X.shape[0], -1, X.shape[1], X.shape[2]),
                ohe=False,
                allow_N=False,
                only_warn=only_warn,
            )
            n_shuffles = references.shape[1]

        n, z = X.shape[0] * n_shuffles, 0

        for i in trange(n, disable=not verbose):
            Xi.append(i // n_shuffles)
            rj.append(i % n_shuffles)

            if len(Xi) == batch_size or i == (n - 1):
                X_batch = X[Xi].cpu().type(dtype)
                args_batch = None if args is None else tuple(
                    a[Xi].to(device).type(dtype) for a in args
                )

                if isinstance(references, torch.Tensor):
                    reference_batch = references[Xi, rj]
                else:
                    if random_state is None:
                        reference_batch = references(X_batch, n=1)[:, 0]
                    else:
                        reference_batch = torch.cat(
                            [
                                references(
                                    X_batch[j : j + 1],
                                    n=1,
                                    random_state=random_state + rj[j],
                                )[:, 0]
                                for j in range(len(X_batch))
                            ]
                        )

                X_batch = X_batch.to(device).type(dtype).requires_grad_()
                reference_batch = (
                    reference_batch.to(device).type(dtype).requires_grad_()
                )

                try:
                    model_inputs = torch.cat([X_batch, reference_batch])

                    if use_autocast:
                        autocast_ctx = torch.autocast(
                            device_type=device.type, dtype=dtype
                        )
                    else:
                        autocast_ctx = contextlib.nullcontext()

                    with torch.autograd.set_grad_enabled(True):
                        with autocast_ctx:
                            if args_batch is not None:
                                repeated_args = (
                                    torch.cat([arg, arg]) for arg in args_batch
                                )
                                y = model(model_inputs, *repeated_args)[:, target]
                            else:
                                y = model(model_inputs)[:, target]

                            multipliers = torch.autograd.grad(y.sum(), X_batch)[0]

                    output_diff = torch.sub(*torch.chunk(y, 2))
                    input_diff = torch.sum(
                        (X_batch - reference_batch) * multipliers, dim=(1, 2)
                    )
                    convergence_deltas = abs(output_diff - input_diff)

                    if torch.any(convergence_deltas > warning_threshold):
                        warnings.warn(
                            "Convergence deltas too high: "
                            + str(convergence_deltas),
                            RuntimeWarning,
                        )

                    if print_convergence_deltas:
                        print(convergence_deltas)

                except Exception as error:
                    model.apply(_clear_hooks)
                    raise error

                if raw_outputs is False:
                    multipliers = hypothetical_attributions(
                        (multipliers,), (X_batch,), (reference_batch,)
                    )[0]

                attr_.extend(list(multipliers.cpu().detach()))

                while len(attr_) >= n_shuffles:
                    attr_chunk = torch.stack(attr_[:n_shuffles])

                    if raw_outputs is False:
                        attr_chunk = attr_chunk.mean(dim=0)
                        if not hypothetical:
                            attr_chunk *= X[z].cpu()

                    attributions.append(attr_chunk)
                    attr_ = attr_[n_shuffles:]
                    z += 1

                if return_references:
                    references_.extend(list(reference_batch.cpu().detach()))

                Xi, rj = [], []

        attributions = torch.stack(attributions)

        if return_references:
            references_ = torch.cat(references_).reshape(
                X.shape[0], n_shuffles, *X.shape[1:]
            )
            return AttributionReferencesResult(
                attributions=attributions, references=references_
            )
        return attributions
    finally:
        model.apply(_clear_hooks)
        for module in model.modules():
            if hasattr(module, "_NON_LINEAR_OPS"):
                del module._NON_LINEAR_OPS

        if was_training:
            model.train()
        if orig_device is not None and orig_device != device:
            model.to(orig_device)


def deep_lift_shap(
    model,
    X,
    args=None,
    target=0,
    batch_size=32,
    references=dinucleotide_shuffle,
    n_shuffles=20,
    return_references=False,
    hypothetical=False,
    warning_threshold=0.001,
    additional_nonlinear_ops=None,
    print_convergence_deltas=False,
    raw_outputs=False,
    only_warn=False,
    dtype=None,
    device="cuda",
    random_state=None,
    verbose=False,
):
    """BPNet-lite DeepLIFT/SHAP wrapper that accepts soft tensor references."""

    bpnet_ops = {
        _ProfileLogitScaling: _nonlinear,
        _Log: _nonlinear,
        _Exp: _nonlinear,
    }
    if additional_nonlinear_ops is not None:
        bpnet_ops.update(additional_nonlinear_ops)

    return _deep_lift_shap_no_reference_ohe_check(
        model=model,
        X=X,
        args=args,
        target=target,
        batch_size=batch_size,
        references=references,
        n_shuffles=n_shuffles,
        return_references=return_references,
        hypothetical=hypothetical,
        warning_threshold=warning_threshold,
        additional_nonlinear_ops=bpnet_ops,
        print_convergence_deltas=print_convergence_deltas,
        raw_outputs=raw_outputs,
        only_warn=only_warn,
        dtype=dtype,
        device=device,
        random_state=random_state,
        verbose=verbose,
    )

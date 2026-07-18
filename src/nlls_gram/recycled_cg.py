# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forked conjugate-gradient core from ``jax.scipy.sparse.linalg``.

``recycled_cg`` is a drop-in replacement for ``jax.scipy.sparse.linalg.cg``.
Only the Krylov loop is vendored here (from the installed jax's
``jax/_src/scipy/sparse/linalg.py``) -- it is the extension point for
deflated/recycled CG across LM steps. The surrounding plumbing (``x0``/
``maxiter`` normalization and the ``lax.custom_linear_solve`` wrapper that
provides implicit derivatives) is imported from ``jax._src`` so upstream
improvements carry over; a jax release that moves those internals will fail
loudly at import time rather than silently diverge.
"""

import jax.numpy as jnp
from jax import lax
from jax._src.scipy.sparse.linalg import (
    _add,
    _identity,
    _isolve,
    _mul,
    _sub,
    _vdot_real_tree,
)
from jax.tree_util import tree_leaves


def _recycled_cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):
    # Verbatim fork of jax's _cg_solve, modulo formatting and public-API
    # imports (jnp.result_type for dtypes.result_type); keep the structure
    # diffable against upstream. Recycling state (deflation basis, Lanczos
    # harvest buffers) will thread through this loop's carry.

    # tolerance handling uses the "non-legacy" behavior of
    # scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # preconditioned CG:
    # en.wikipedia.org/wiki/Conjugate_gradient_method (preconditioned variant)

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1

    r0 = _sub(b, A(x0))
    p0 = z0 = M(r0)
    dtype = jnp.result_type(*tree_leaves(p0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    initial_value = (x0, r0, gamma0, p0, 0)

    x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final


def recycled_cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """Drop-in fork of :func:`jax.scipy.sparse.linalg.cg`.

    Semantics are identical to the upstream solver: ``A`` is a hermitian
    positive-definite matvec callable (or matrix), convergence is
    ``norm(residual) <= max(tol * norm(b), atol)``, ``M`` approximates
    ``A^{-1}``, ``x0`` seeds the iteration, and derivatives are implicit
    (another CG solve through ``lax.custom_linear_solve``) rather than
    differentiated through the iterations. Returns ``(x, info)`` with
    ``info=None``, matching upstream.

    The Krylov loop is vendored in this package as the extension point for
    Krylov-subspace recycling across the LM solver's successive dual solves;
    with no recycling state it reproduces upstream CG exactly.
    """
    return _isolve(
        _recycled_cg_solve,
        A=A,
        b=b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        check_symmetric=True,
    )

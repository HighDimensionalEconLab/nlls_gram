"""``LevenbergMarquardtBase`` -- everything the two solvers do identically.

The subclasses supply the objective and its algebra (``init``, ``update``,
``_initial_info``, ``_converged``, ``_cast_state``, ``_cold_state``,
``_ranking_objective``, ``_ad_x_tangent``); the loop driving, callback action
handling, Jacobian assembly, static-key identity, and the ``custom_jvp``
implicit-AD wrapper live here.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from nlls_gram.lm_types import (
    LMHyperparams,
    LMSolveAction,
    LMStatus,
    SolverContext,
    _damping_floor,
)
from nlls_gram.multi_start import (
    MultiStart,
    _accept_converged,
    _accept_converged_or_max_steps,
    _check_drawn_types,
    _multi_start_parallel_jit,
    _multi_start_python_impl,
    _multi_start_sequential_jit,
)
from nlls_gram.preconditioners import Preconditioner
from nlls_gram.solve_loop import _solve_loop_jit, _solve_python_impl
from nlls_gram.utilities import (
    _hashable_hook,
    _mask_tangent_tree,
    _tree_changed,
    _where_tree,
    _zero_tangent_leaf,
)


class LevenbergMarquardtBase:
    # Value-based identity: the jitted solve loop marks the solver itself
    # static, so equal-config solvers built around the same residual share the
    # compiled loop across instances. Subclasses set _static_key/_static_hash
    # in __init__ from their constructor arguments.
    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._static_key == other._static_key

    def __hash__(self):
        return self._static_hash

    def _validate_configuration(self, linear_solver, ad_solver, penalized):
        """Reject a config in a role it cannot fill, at construction.

        Without this an unsupported combination resolves to a DIFFERENT
        algorithm -- ``ad_solver=QR()`` quietly running dense Cholesky, or the
        ridge solver handing its penalized subproblem to a dual form that
        never sees the penalty rows.
        """
        role = [(linear_solver, "linear_solver", "supports_forward")]
        if ad_solver is not None:
            role.append((ad_solver, "ad_solver", "supports_ad"))
        for config, keyword, attribute in role:
            if not getattr(config, attribute):
                raise ValueError(
                    f"{config!r} cannot serve as {keyword}: it does not "
                    f"implement that role"
                )
            if penalized and not config.supports_penalty:
                raise ValueError(
                    f"{config!r} cannot serve as {keyword} for the ridge "
                    "objective: its operator never sees the penalty rows"
                )
        if self.jacobian_mode not in ("auto", "fwd", "rev"):
            raise ValueError(f"unknown jacobian_mode: {self.jacobian_mode}")

    def _block_sizes(self, theta_size):
        # The free-block size is inferred from the flattened iterate: the
        # metric covers the leading metric.size coordinates, the rest is free.
        n_m = self.metric.size
        if n_m > theta_size:
            raise ValueError(
                f"the metric covers {n_m} leading coordinates but x flattens "
                f"to only {theta_size}; the free block is len(x) - metric.size "
                "and must be nonnegative"
            )
        return n_m, theta_size - n_m

    def _cold_state(self, lm_state):
        # Drawn multi-start lanes must not reuse caches or hook state built at
        # another (x, args); damping, ridge, and hyper stay inherited from the
        # caller's initial state.
        updates = {}
        for flag in ("jacobian_valid", "metric_valid", "precond_valid"):
            if getattr(lm_state, flag) is not None:
                updates[flag] = jnp.zeros_like(getattr(lm_state, flag))
        if lm_state.solver_cache is not None:
            updates["solver_cache"] = jax.tree.map(
                jnp.zeros_like, lm_state.solver_cache
            )
        return dataclasses.replace(lm_state, **updates) if updates else lm_state

    def _resolve_jacobian_mode(self, m, n):
        # "auto" vmaps the identity basis of the SMALL side: n forward-mode
        # columns when the system is tall or square (n <= m), m reverse-mode
        # rows only when strictly fat.
        if self.jacobian_mode != "auto":
            return self.jacobian_mode
        return "fwd" if n <= m else "rev"

    def _assemble_jt(self, jvp_fn, theta, resid):
        if self._resolve_jacobian_mode(resid.shape[0], theta.shape[0]) == "fwd":
            parameter_basis = jnp.eye(theta.shape[0], dtype=theta.dtype)
            return jax.vmap(jvp_fn)(parameter_basis)
        transpose_fn = jax.linear_transpose(jvp_fn, theta)
        residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
        return jax.vmap(lambda cotangent: transpose_fn(cotangent)[0])(residual_basis).T

    def _dense_resid_jt_aux(self, residual_flat, theta):
        if self.has_aux:
            resid, jvp_fn, aux = jax.linearize(residual_flat, theta, has_aux=True)
        else:
            resid, jvp_fn = jax.linearize(residual_flat, theta)
            aux = None
        return resid, self._assemble_jt(jvp_fn, theta, resid), aux

    def _residual_and_aux(self, x, args, p):
        if self.has_aux:
            value, aux = self.residual_fn(x, args, p)
            return jnp.ravel(value), aux
        return jnp.ravel(self.residual_fn(x, args, p)), None

    def _check_residual_args(self, args, p):
        # Silently dropping args/p a residual never sees would, in particular,
        # make the implicit derivative with respect to p a silent zero.
        if args is not None and self.residual_arity < 2:
            raise ValueError(
                "args was passed but residual_fn takes only (x); "
                "use residual_fn(x, args)"
            )
        if p is not None and self.residual_arity < 3:
            raise ValueError(
                "p was passed but residual_fn takes no p argument; "
                "use residual_fn(x, args, p)"
            )

    def hyperparams(self, dtype=None):
        """``LMHyperparams`` built from the constructor values."""
        iterative_tol = self.iterative_tol
        if iterative_tol is None:
            # NaN marks "resolve against the problem dtype"; _cast_hyper does
            # it once the residual dtype is known. Resolving here would bake in
            # the JAX default float, which is wrong for a float32 problem under
            # enabled x64.
            if dtype is None:
                iterative_tol = jnp.nan
            else:
                iterative_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return LMHyperparams(
            jnp.asarray(self.damping_decrease, dtype=dtype),
            jnp.asarray(self.damping_increase, dtype=dtype),
            _damping_floor(self.min_damping, dtype),
            None
            if self.max_damping is None
            else jnp.asarray(self.max_damping, dtype=dtype),
            jnp.asarray(self.geodesic_acceptance_ratio, dtype=dtype),
            jnp.asarray(iterative_tol, dtype=dtype),
            jnp.asarray(self.iterative_atol, dtype=dtype),
            None
            if self.iterative_maxiter is None
            else jnp.asarray(self.iterative_maxiter, dtype=jnp.int32),
        )

    def _block_sizes(self, theta_size):
        # The free-block size is inferred from the flattened iterate: the
        # metric covers the leading metric.size coordinates, the rest is free.
        n_f = theta_size - self.metric.size
        if n_f < 0:
            raise ValueError(
                f"the metric covers {self.metric.size} leading coordinates "
                f"but x flattens to only {theta_size}; the free block is "
                "len(x) - metric.size and must be nonnegative"
            )
        return self.metric.size, n_f

    # The solver-internal extension F_bar = blockdiag(F, sqrt(free_scale) I):
    # the metric's factor op on the metric block, a scalar on the free block.
    # Applied to vectors or leading-axis-batched matrices; F_bar itself is
    # never materialized, and the free block drops out entirely when it is
    # empty or unscaled.
    def _free_scale(self, v):
        scale = self.metric.free_scale
        return v if scale == 1.0 else v / jnp.sqrt(jnp.asarray(scale, v.dtype))

    def _extended_solve(self, v, ctx):
        n_m = self.metric.size
        if n_m == 0:
            return self._free_scale(v)
        if v.shape[0] == n_m:
            return self.metric.factor_solve(v, ctx)
        return jnp.concatenate(
            [self.metric.factor_solve(v[:n_m], ctx), self._free_scale(v[n_m:])], axis=0
        )

    def _extended_solve_transpose(self, v, ctx):
        n_m = self.metric.size
        if n_m == 0:
            return self._free_scale(v)
        if v.shape[0] == n_m:
            return self.metric.factor_solve_transpose(v, ctx)
        return jnp.concatenate(
            [
                self.metric.factor_solve_transpose(v[:n_m], ctx),
                self._free_scale(v[n_m:]),
            ],
            axis=0,
        )

    def _init_hook_state(self, theta, lm_state, args, p):
        """The metric's and preconditioner's state at ``x0``, valid there, so
        the first update reuses it. The flags stay absent when the hooks are
        stateless."""
        ctx = SolverContext(x=theta, lm_state=lm_state, args=args, p=p)
        valid = jnp.asarray(True, dtype=jnp.bool_)
        hooks = {}
        if self._metric_prepares:
            hooks["metric_state"] = self.metric.prepare(theta, ctx)
            hooks["metric_valid"] = valid
        if self._precond_prepares:
            hooks["precond"] = self.preconditioner.prepare(theta, ctx)
            hooks["precond_valid"] = valid
        return hooks

    def _hook_state(self, theta, lm_state, args, p):
        """The metric's and preconditioner's prepared state for this step:
        reused while still valid (a rejected step left ``x`` in place, or the
        hook declined to rebuild), rebuilt from the live iterate otherwise."""
        bare = SolverContext(x=theta, lm_state=lm_state, args=args, p=p)
        metric_state = precond_state = None
        if self._metric_prepares:
            metric_state = jax.lax.cond(
                lm_state.metric_valid | ~jnp.asarray(self.metric.rebuild(bare)),
                lambda _: lm_state.metric_state,
                lambda _: self.metric.prepare(theta, bare),
                operand=None,
            )
        if self._precond_prepares:
            precond_state = jax.lax.cond(
                lm_state.precond_valid
                | ~jnp.asarray(self.preconditioner.rebuild(bare)),
                lambda _: lm_state.precond,
                lambda _: self.preconditioner.prepare(theta, bare),
                operand=None,
            )
        return metric_state, precond_state

    def _carried_ctx(self, theta, lm_state, args, p):
        return SolverContext(
            x=theta,
            lm_state=lm_state,
            args=args,
            p=p,
            metric_state=lm_state.metric_state,
            preconditioner_state=lm_state.precond,
        )

    def _frozen_ctx(self, theta, lm_state, args, p, preconditioner):
        # Under implicit AD the hooks are FROZEN at the returned solution:
        # prepare runs once there and the state-dependence is not
        # differentiated, the same contract as a fixed metric closing over
        # constants.
        bare = SolverContext(x=theta, lm_state=lm_state, args=args, p=p)
        metric_state = (
            self.metric.prepare(theta, bare) if self._metric_prepares else None
        )
        precond_state = None
        if preconditioner is not None and (
            type(preconditioner).prepare is not Preconditioner.prepare
        ):
            precond_state = preconditioner.prepare(theta, bare)
        return dataclasses.replace(
            bare, metric_state=metric_state, preconditioner_state=precond_state
        )

    def _ad_linearization(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        return theta, unravel, residual, theta_jvp, residual_p_dot

    def _ad_cg_tol(self, dtype):
        if self.ad_solver_tol is not None:
            return jnp.asarray(self.ad_solver_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)

    def _action_or_default(self, action):
        if action is None:
            return LMSolveAction()
        return action

    def _apply_action(self, action, x, lm_state, args, user_state):
        action = self._action_or_default(action)
        # The step's diagnostics and every cache describe the pre-action
        # problem, so they are stale iff the action actually changed the
        # values -- a traced comparison, so a jit-style callback that returns
        # the field every step with unchanged values changes nothing.
        xargs_changed = jnp.asarray(False)
        state_changed = jnp.asarray(False)
        if action.x is not None:
            xargs_changed = xargs_changed | _tree_changed(action.x, x)
            x = action.x
        if action.lm_state is not None:
            previous = lm_state
            lm_state = action.lm_state
            if self.cache_jacobian and lm_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the callback action returned an "
                    "lm_state without the Jacobian cache; use "
                    "dataclasses.replace(ctx.lm_state, ...) to preserve the "
                    "cache fields"
                )
            self._check_action_state(lm_state)
            # Trace-time guard so the hyper contract fails identically with
            # and without jit (jit would reject the carry mismatch anyway).
            previous_hyper = previous.hyper
            if previous_hyper is not None and (
                lm_state.hyper is None
                or jax.tree_util.tree_structure(previous_hyper)
                != jax.tree_util.tree_structure(lm_state.hyper)
                or [leaf.dtype for leaf in jax.tree_util.tree_leaves(previous_hyper)]
                != [leaf.dtype for leaf in jax.tree_util.tree_leaves(lm_state.hyper)]
            ):
                raise ValueError(
                    "the callback action changed the structure or dtypes of "
                    "lm_state.hyper; reset values with "
                    "dataclasses.replace(ctx.lm_state.hyper, ...) using arrays "
                    "of the same dtype — a knob constructed as None cannot be "
                    "enabled mid-solve"
                )
            lm_state, state_changed = self._apply_action_state(lm_state, previous)
        if action.args is not None:
            xargs_changed = xargs_changed | _tree_changed(action.args, args)
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        problem_changed = xargs_changed | state_changed
        if action.x is not None or action.args is not None:
            # Everything prepared at the pre-action (x, args) is stale once
            # either moves. The metric matters most: it DEFINES the subproblem,
            # so reusing a factor built at the old iterate would silently solve
            # a different problem in a different geometry.
            stale = {}
            for flag in ("jacobian_valid", "metric_valid", "precond_valid"):
                carried = getattr(lm_state, flag)
                if carried is not None:
                    stale[flag] = carried & ~xargs_changed
            if stale:
                lm_state = dataclasses.replace(lm_state, **stale)
        touched = (
            action.x is not None
            or action.args is not None
            or action.lm_state is not None
        )
        lm_state = self._invalidate_caches(lm_state, action, touched, problem_changed)
        return action, x, lm_state, args, user_state, problem_changed

    # Subclass hooks for the callback-action path. The defaults are inert.
    def _check_action_state(self, lm_state):
        pass

    def _apply_action_state(self, lm_state, previous):
        return lm_state, jnp.asarray(False)

    def _invalidate_caches(self, lm_state, action, touched, problem_changed):
        if touched and lm_state.solver_cache is not None:
            cache = lm_state.solver_cache
            lm_state = dataclasses.replace(
                lm_state,
                solver_cache=dataclasses.replace(
                    cache, valid=cache.valid & ~problem_changed
                ),
            )
        return lm_state

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        lm_state=None,
        max_steps=256,
        max_steps_is_success=True,
        atol=0.0,
        gtol=0.0,
        xtol=0.0,
        callback=None,
        user_state=None,
        save_steps=False,
        multi_start=None,
        jit=True,
    ):
        """Run repeated LM updates until a stopping rule fires.

        Parameters are ``update``'s plus loop controls. ``max_steps`` is always
        enforced; ``max_steps_is_success=True`` (the default) treats
        ``LMStatus.MAX_STEPS`` as usable for implicit AD and for the default
        multi-start acceptance, while keeping the status for diagnostics.

        ``atol``/``gtol``/``xtol`` bound the residual norm, the stationarity
        residual ``info.grad_norm``, and an accepted step's ``info.step_norm``;
        ``0`` disables a check and every firing rule reports
        ``LMStatus.CONVERGED``. How the three combine is the solver's own
        contract -- see each subclass.

        ``callback`` receives an ``LMSolveContext`` after each step and may
        return an ``LMSolveAction`` to stop or to override x/lm_state/args/
        user_state; ``p`` is passed through but cannot be replaced. A callback
        that installs an invalid ``x`` or ``args`` must also stop with a failed
        status.

        ``save_steps=True`` records the iterate history on the result:
        ``x_history`` stacks ``x0`` and every kept post-step iterate along a
        ``(max_steps + 1)`` leading axis (rows beyond ``steps`` are zero
        padding), plus the row-aligned ``args_history`` and, with ``has_aux``,
        ``aux_history``. The buffers are differentiation-inert and make the
        jitted loop retrace when ``max_steps`` changes.

        ``multi_start`` (a ``MultiStart``) retries or parallelizes over fresh
        initial conditions and returns the single best result, with
        diagnostics on ``result.multi_start``.

        Implicit AD uses ``CONVERGED``, and ``MAX_STEPS`` when
        ``max_steps_is_success=True``; every failed status receives zero
        tangents for ``result.x`` and ``result.aux``, with the failed lane's
        linear tangent program evaluated at differentiation-inert copies of the
        original ``(x0, args, p)``.

        ONLY ``result.x``, ``result.aux``, and ``result.p`` carry tangents.
        Everything else -- ``info`` (including ``info.loss``), ``lm_state``,
        ``steps``, the histories, and the multi-start diagnostics -- is
        differentiation-inert and gets EXACT ZERO, since damping, step norms,
        and iteration counts are artifacts of the path rather than properties
        of the root. Differentiating a loss therefore means recomputing it
        from ``result.x``, not reading ``result.info.loss``::

            jax.grad(lambda p: objective(solver.solve(x0, p=p).x, p))(p)
        """
        self._check_residual_args(args, p)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        # Tolerances are traced data inside the loop, so vmapped/traced values
        # skip the concrete-only sign validation.
        if not isinstance(atol, jax.core.Tracer) and atol < 0:
            raise ValueError("atol must be nonnegative")
        if not isinstance(gtol, jax.core.Tracer) and gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if not isinstance(xtol, jax.core.Tracer) and xtol < 0:
            raise ValueError("xtol must be nonnegative")
        self._validate_tolerances(atol, gtol, xtol)
        callback = _hashable_hook(callback)
        lm_state = self._solve_lm_state(x0, args, p, lm_state)
        if lm_state.hyper is None:
            lm_state = dataclasses.replace(lm_state, hyper=self.hyperparams())
        history_len = max_steps + 1 if save_steps else None

        if multi_start is not None:
            num_starts = multi_start.num_starts
            draw = _hashable_hook(multi_start.draw if num_starts > 1 else None)
            default_accept = (
                _accept_converged_or_max_steps
                if max_steps_is_success
                else _accept_converged
            )
            accept = _hashable_hook(
                default_accept if multi_start.accept is None else multi_start.accept
            )
            parallel = multi_start.parallel and num_starts > 1
            if draw is not None and jit:
                drawn = jax.eval_shape(draw, multi_start.key, x0, args)
                _check_drawn_types(x0, args, drawn)

            @jax.custom_jvp
            def solve_multi_start_with_ad_p(
                x, lm_state, args, p, user_state, key, max_steps, atol, gtol, xtol
            ):
                return self._multi_start_impl(
                    x,
                    lm_state,
                    args,
                    p,
                    user_state,
                    key,
                    history_len,
                    max_steps,
                    atol,
                    gtol,
                    xtol,
                    callback,
                    jit,
                    num_starts,
                    draw,
                    accept,
                    parallel,
                )

            @solve_multi_start_with_ad_p.defjvp
            def solve_multi_start_with_ad_p_jvp(primals, tangents):
                p_dot = tangents[3]
                result = solve_multi_start_with_ad_p(*primals)
                point = self._initial_ad_point(
                    primals[0], primals[1], primals[2], primals[3]
                )
                return result, self._ad_result_tangent(
                    result, p_dot, point, max_steps_is_success
                )

            return solve_multi_start_with_ad_p(
                x0,
                lm_state,
                args,
                p,
                user_state,
                multi_start.key,
                max_steps,
                atol,
                gtol,
                xtol,
            )

        @jax.custom_jvp
        def solve_with_ad_p(
            x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        ):
            return self._solve_impl(
                x,
                lm_state,
                args,
                p,
                user_state,
                history_len,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
                jit,
            )

        @solve_with_ad_p.defjvp
        def solve_with_ad_p_jvp(primals, tangents):
            p_dot = tangents[3]
            result = solve_with_ad_p(*primals)
            point = self._initial_ad_point(
                primals[0], primals[1], primals[2], primals[3]
            )
            return result, self._ad_result_tangent(
                result, p_dot, point, max_steps_is_success
            )

        return solve_with_ad_p(
            x0, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        )

    # Subclass hooks for solve. The defaults suit a solver with no extra
    # state contract beyond the shared LMState.
    def _validate_tolerances(self, atol, gtol, xtol):
        pass

    def _solve_lm_state(self, x0, args, p, lm_state):
        return self.init(x0, args, p=p) if lm_state is None else lm_state

    def _initial_ad_point(self, x, lm_state, args, p):
        return (x, args, p)

    def _solve_impl(
        self,
        x,
        lm_state,
        args,
        p,
        user_state,
        history_len,
        max_steps,
        atol,
        gtol,
        xtol,
        callback,
        jit,
    ):
        driver = _solve_loop_jit if jit else _solve_python_impl
        return driver(
            self,
            x,
            lm_state,
            args,
            p,
            user_state,
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )

    def _multi_start_impl(
        self,
        x,
        lm_state,
        args,
        p,
        user_state,
        key,
        history_len,
        max_steps,
        atol,
        gtol,
        xtol,
        callback,
        jit,
        num_starts,
        draw,
        accept,
        parallel,
    ):
        if not jit:
            return _multi_start_python_impl(
                self,
                x,
                lm_state,
                args,
                p,
                user_state,
                key,
                history_len,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
                num_starts,
                draw,
                accept,
                parallel,
            )
        if parallel:
            return _multi_start_parallel_jit(
                self,
                x,
                lm_state,
                args,
                p,
                user_state,
                key,
                history_len,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
                draw,
                accept,
                num_starts,
            )
        return _multi_start_sequential_jit(
            self,
            x,
            lm_state,
            args,
            p,
            user_state,
            key,
            jnp.asarray(num_starts, dtype=jnp.int32),
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
            draw,
            accept,
        )

    def _ad_result_tangent(self, result, p_dot, initial_ad_point, max_steps_is_success):
        # A successful tangent relinearizes at the returned solution. A failed
        # tangent uses the differentiation-inert original initial point, so the
        # linear tangent program stays finite under vmap. Everything except x,
        # p, and aux is bookkeeping with zero tangents.
        initial_x, initial_args, initial_p = jax.tree.map(
            jax.lax.stop_gradient, initial_ad_point[:3]
        )
        ad_success = result.status == LMStatus.CONVERGED
        if max_steps_is_success:
            ad_success = ad_success | (result.status == LMStatus.MAX_STEPS)
        ad_x = _where_tree(ad_success, result.x, initial_x)
        ad_args = _where_tree(ad_success, result.args, initial_args)
        ad_p = _where_tree(ad_success, result.p, initial_p)
        ad_p_dot = _mask_tangent_tree(ad_success, p_dot)
        zero_result = jax.tree.map(_zero_tangent_leaf, result)
        x_dot = self._ad_x_tangent(
            ad_x, ad_args, ad_p, ad_p_dot, result, ad_success, initial_ad_point
        )
        x_dot = _where_tree(ad_success, x_dot, zero_result.x)
        aux_dot = zero_result.aux
        if self.has_aux and ad_p is not None:
            # aux depends on p directly and through the solution x*(p).
            def aux_at_solution(x_value, p_value):
                return self.residual_fn(x_value, ad_args, p_value)[1]

            aux_dot = jax.jvp(aux_at_solution, (ad_x, ad_p), (x_dot, ad_p_dot))[1]
            aux_dot = _where_tree(ad_success, aux_dot, zero_result.aux)
        return dataclasses.replace(zero_result, x=x_dot, p=p_dot, aux=aux_dot)


# Re-exported for the solvers' isinstance-free solve signature.
__all__ = ["LevenbergMarquardtBase", "MultiStart"]

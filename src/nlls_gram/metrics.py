import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def metric_callbacks_from_cholesky(L, *, lower=True):
    """Return metric callbacks for a dense Cholesky metric factor.

    Parameters
    ----------
    L:
        Cholesky factor of the metric matrix. For ``lower=True``, this helper
        treats ``M = L @ L.T``.
    lower:
        Whether ``L`` is lower triangular.

    Returns
    -------
    dict
        Callback dictionary with ``metric_solve``, ``metric_norm``,
        ``metric_inv_sqrt``, and ``metric_inv_sqrt_transpose``.
    """
    if not lower:
        raise NotImplementedError(
            "metric_callbacks_from_cholesky currently expects lower=True"
        )

    def metric_solve(x):
        y = jsp_linalg.solve_triangular(L, x, lower=True)
        return jsp_linalg.solve_triangular(L.T, y, lower=False)

    def metric_norm(x):
        return jnp.linalg.norm(L.T @ x)

    def metric_inv_sqrt(x):
        return jsp_linalg.solve_triangular(L.T, x, lower=False)

    def metric_inv_sqrt_transpose(x):
        return jsp_linalg.solve_triangular(L, x, lower=True)

    return {
        "metric_solve": metric_solve,
        "metric_norm": metric_norm,
        "metric_inv_sqrt": metric_inv_sqrt,
        "metric_inv_sqrt_transpose": metric_inv_sqrt_transpose,
    }

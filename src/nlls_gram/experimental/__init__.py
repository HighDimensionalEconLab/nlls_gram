"""Experimental extras: not part of the supported API, not documented, and
free to change or disappear. Import explicitly::

    from nlls_gram.experimental import StateSpaceMetric, matern_state_space
"""

from nlls_gram.experimental.state_space_metric import (
    StateSpaceMetric,
    matern_state_space,
)

__all__ = ["StateSpaceMetric", "matern_state_space"]

import argparse
import json
import platform
import statistics
import subprocess
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.conftest import _benchmark_supports_float64
from benchmarks.test_implicit_ad_benchmark import (
    SOLVER_CASES,
    TRANSFORMS,
    _make_transformed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=None,
        choices=["float32", "float64"],
    )
    args = parser.parse_args()
    if args.dtypes is None:
        args.dtypes = (
            ["float32", "float64"] if _benchmark_supports_float64() else ["float32"]
        )

    dtype_map = {"float32": jnp.float32, "float64": jnp.float64}
    timings = {}
    for dtype_name in args.dtypes:
        dtype = dtype_map[dtype_name]
        with jax.enable_x64(dtype == jnp.float64):
            for transform in TRANSFORMS:
                for case in SOLVER_CASES:
                    samples = []
                    for _ in range(args.repeat):
                        jax.clear_caches()
                        transformed, parameter = _make_transformed(
                            case, transform, dtype=dtype
                        )
                        start = timeit.default_timer()
                        compiled = transformed.lower(parameter).compile()
                        elapsed = timeit.default_timer() - start
                        value = compiled(parameter)
                        jax.block_until_ready(value)
                        samples.append(elapsed)
                    timings[f"{dtype_name}-{transform}-{case}"] = {
                        "median_seconds": statistics.median(samples),
                        "samples_seconds": samples,
                    }

    payload = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "host": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": jax.default_backend(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "repeat": args.repeat,
        "timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

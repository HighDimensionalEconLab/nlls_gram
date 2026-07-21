"""Cold compilation timings for the implicit-AD benchmark matrix."""

import argparse
import json
import platform
import statistics
import subprocess
import timeit
from pathlib import Path

import jax

from benchmarks.test_implicit_ad_benchmark import _make_transformed

CASES = ("direct", "direct_aux", "metric_factory", "vmapped")
TRANSFORMS = ("jvp", "vjp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    timings = {}
    for transform in TRANSFORMS:
        for case in CASES:
            samples = []
            for _ in range(args.repeat):
                jax.clear_caches()
                transformed, parameter = _make_transformed(case, transform)
                start = timeit.default_timer()
                compiled = transformed.lower(parameter).compile()
                elapsed = timeit.default_timer() - start
                value = compiled(parameter)
                jax.block_until_ready(value)
                samples.append(elapsed)
            timings[f"{transform}-{case}"] = {
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

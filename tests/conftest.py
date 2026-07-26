import jax

# The tests build their own dense references (`selection.T @ shifted @
# selection`, `F_bar @ v`) and compare them against solver output. The package
# pins its own products to HIGHEST, so without this the reference side would
# still come from TF32 tensor cores on an Ampere GPU and the two would disagree
# at ~1e-3 -- a comparison failing on the reference, not on the solver.
# A no-op on CPU.
jax.config.update("jax_default_matmul_precision", "highest")

# JustInTimeDecoding
Data-generation used for simulating the JIT decoder in a cubic lattice, including twisted errors induced by the nonabelian stabilizers of the twisted quantum double model and the heralding heuristic proposed in arxiv/2604.02033.  

## Module layout

- geometry.py: lattice indexing, time-depth rule, last-step measured-edge mask,
  canonical output filename builder.
- lattice.py: incidence matrix construction, neighbor-edge lookup,
  vectorization masks, local edge-mask helper.
- decoder.py: logical-error checks and JIT decoding protocol.
- twisted.py: twisted-Z error generation and loop-closing matching builder.
- runner.py: high-level simulation entry points:
  - run_full_simulation
  - run_x_only_simulation
- multilayer.py: N-layer generalization where every layer is corrected just in
  time, interleaved per time step; flux loops of layer i delegate errors into
  layer i+1. Per-layer delegated-error generation and decoder-side accounting
  (heralded 0-weight links) are passed in as functions (LayerSpec); defaults
  reproduce the twisted-error behavior of the two-layer code.
  - run_multilayer_simulation (JIT cascade + global-cascade baseline)
- cluster/: Slurm array worker and collector for multilayer sweeps
  (multilayer_worker.py, multilayer_collect.py), driven by multilayer.slurm.sh.
- cli.py: command-line wrapper preserving legacy argument order.

## Quick usage

```bash
python -m shared_simulation.cli results/xz_errs 9 0.02 0.02 10000 0
```

Or from Python:

```python
from shared_simulation.runner import run_full_simulation

counters = run_full_simulation(
    linear_size=9,
    px=0.02,
    pz=0.02,
    repetitions=10000,
    output_dir="results/xz_errs",
    boundary="OBC",
    run_id=0,
    use_jit=True,
)
print(counters)
```

## Multi-layer JIT

```python
from shared_simulation.multilayer import (
    make_default_layer_specs,
    run_multilayer_simulation,
)

result = run_multilayer_simulation(
    linear_size=9,
    layer_specs=make_default_layer_specs([0.02, 0.02, 0.02]),
    repetitions=10000,
    output_dir="results/multilayer",
    boundary="OBC",
    run_id=0,
)
print(result)
```

Custom physics per layer: pass your own `generate_delegated_errors` /
`account_delegated_errors` callables in each `LayerSpec` (see the contracts in
the multilayer.py module docstring).

Cluster sweep: edit the configuration block in `multilayer.slurm.sh`
(200-task array), then `sbatch multilayer.slurm.sh` from the repository root
and aggregate with `python cluster/multilayer_collect.py` once the array
finishes.

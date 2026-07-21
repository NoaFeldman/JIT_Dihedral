"""Multi-layer just-in-time decoding pipeline.

Generalizes the two-layer protocol (JIT X layer + globally corrected twisted-Z
layer, arxiv/2604.02033) to a stack of N layers that are *all* corrected just
in time. Flux loops in layer i delegate errors into layer i+1 (the multi-layer
analogue of the 'twisted' errors in twisted.py). At every time step t the
protocol:

1. corrects the time-t sublattice of layer 0 (standard JIT step);
2. generates the delegated errors that layer 0's flux has just created in
   layer 1 (simulation ground truth, XOR-ed into layer 1's noise), and
   separately lets layer 1's decoder *account* for the delegated errors it can
   infer, by setting the matching weight of the heralded links to 0;
3. corrects the time-t sublattice of layer 1 with those weights;
4. repeats (2)-(3) down the stack for layers 2..N-1, then moves to t+1.

Both the generation of delegated errors and their decoder-side accounting may
differ per layer, so each LayerSpec carries the two functions as inputs
(generate_delegated_errors / account_delegated_errors). Defaults reproducing
the twisted-error behavior of the existing two-layer code are provided
(twisted_delegated_generator / twisted_delegated_accountant).

Contracts for the per-layer functions (see also ParentLayerView):

- generate_delegated_errors(context, time_step, parent) -> {color: uint8 array}
  Returns the delegated errors *newly created at this time step* in the
  receiving layer, given the parent layer's state after its time_step decode.
  The pipeline XOR-accumulates the returned configuration into the layer's
  noise, so the function must not re-emit contributions from earlier steps.
  `parent.residual` (ground-truth flux loops, noise ^ correction) is available
  here because generation is a simulation-side operation.

- account_delegated_errors(context, time_step, parent) -> {color: uint8 mask}
  Returns per-color heralded-link masks; the layer's matchings for this step
  are built with weight 0 on the heralded links. This models the decoder, so
  `parent.residual` is withheld (None): only the parent's corrections are
  decoder-visible information.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .decoder import is_logical_error
from .geometry import (
    DIMENSIONS,
    get_time_depth,
    last_time_step_measurement_edges,
    vertex_index,
)
from .lattice import (
    build_edge_endpoints,
    build_incidence_matrix,
    build_neighbor_edge_lookup,
    precompute_twist_masks,
)
from .twisted import generate_twisted_z_errors

COLORS = ["b", "g", "r"]

ColorConfig = Dict[str, np.ndarray]


@dataclass
class ParentLayerView:
    """What a layer exposes to the layer above it at one time step.

    correction: parent's mod-2 JIT correction after its decode at the current
        time step (full-lattice arrays, per color).
    previous_correction: parent's mod-2 correction before the current step's
        decode; (correction ^ previous_correction) is the loop configuration
        the parent's decoder added in this step.
    residual: ground-truth flux loops (noise ^ correction). Only populated for
        generation; accounting receives None here, since a real decoder cannot
        observe the residual.
    """

    correction: ColorConfig
    previous_correction: ColorConfig
    residual: Optional[ColorConfig]


DelegatedErrorGenerator = Callable[["MultiLayerContext", int, ParentLayerView], ColorConfig]
DelegatedErrorAccountant = Callable[["MultiLayerContext", int, ParentLayerView], ColorConfig]


@dataclass
class LayerSpec:
    """Per-layer configuration of the multi-layer JIT stack.

    Layer 0 has no parent, so its two callbacks are ignored and may be None.
    For layers i > 0, generate_delegated_errors defines the physical delegated
    errors induced by layer i-1 and account_delegated_errors defines the
    heralding used by layer i's decoder (None disables heralding).
    """

    noise_probability: float
    error_type: Literal["x", "z"] = "z"
    generate_delegated_errors: Optional[DelegatedErrorGenerator] = None
    account_delegated_errors: Optional[DelegatedErrorAccountant] = None


@dataclass
class MultiLayerContext:
    """Precomputed geometry shared by every layer and repetition."""

    linear_size: int
    time_depth: int
    boundary: str
    num_edges: int
    full_incidence: csc_matrix
    full_matching: Matching
    prefix_incidences: List[csc_matrix]
    prefix_matchings: List[Matching]
    edge_lookup: dict
    twist_masks: tuple
    edge_endpoints: np.ndarray
    last_measured_edges: np.ndarray = field(repr=False, default=None)


def build_multilayer_context(linear_size: int, boundary: str = "OBC") -> MultiLayerContext:
    """Build the shared geometry/matching context for a multi-layer run."""
    time_depth = get_time_depth(linear_size, boundary)
    full_incidence = build_incidence_matrix(linear_size, time_depth, boundary)
    prefix_incidences = [
        build_incidence_matrix(linear_size, t_idx + 1, open_end_node=(t_idx < time_depth - 1))
        for t_idx in range(time_depth)
    ]
    return MultiLayerContext(
        linear_size=linear_size,
        time_depth=time_depth,
        boundary=boundary,
        num_edges=DIMENSIONS * linear_size**2 * time_depth,
        full_incidence=full_incidence,
        full_matching=Matching(full_incidence),
        prefix_incidences=prefix_incidences,
        prefix_matchings=[Matching(prefix_h) for prefix_h in prefix_incidences],
        edge_lookup=build_neighbor_edge_lookup(linear_size, time_depth),
        twist_masks=precompute_twist_masks(linear_size, time_depth),
        edge_endpoints=build_edge_endpoints(full_incidence),
        last_measured_edges=last_time_step_measurement_edges(linear_size, time_depth),
    )


def twisted_delegated_generator(
    context: MultiLayerContext,
    time_step: int,
    parent: ParentLayerView,
) -> ColorConfig:
    """Single-time-slice stochastic twisted-error generation.

    Incremental analogue of generate_twisted_z_errors(..., is_full=False): the
    same per-vertex twist rules, applied with probability 1/2 per vertex and
    color, but restricted to the vertices of the slice revealed at this time
    step, so that XOR-accumulating over all steps reproduces the full-lattice
    stochastic generation without re-randomizing earlier slices.

    Frontier caveat: the twist rules at slice t touch edges of slices t-1 and
    t+1; the parent residual on slice t+1 may still change in later JIT steps.
    This mirrors the physical setting where delegated errors are committed
    when the slice-t correction is applied.
    """
    linear_size = context.linear_size
    time_depth = context.time_depth
    edge_lookup = context.edge_lookup
    x_loops = parent.residual
    if x_loops is None:
        raise ValueError("twisted_delegated_generator requires parent.residual.")

    twisted = {color: np.zeros(context.num_edges, dtype=np.uint8) for color in COLORS}
    t_index = time_step
    for vertex_color in COLORS:
        is_twisted = np.random.randint(0, 2, size=(linear_size, linear_size))
        for x_index in range(linear_size):
            for y_index in range(linear_size):
                if not is_twisted[x_index, y_index]:
                    continue
                node = vertex_index(x_index, y_index, t_index, linear_size)
                if vertex_color == "g":
                    edge_fwd = edge_lookup[(node, 1)]
                    edge_bwd = edge_lookup[(node, -1)]
                    twisted["b"][edge_fwd] ^= x_loops["r"][edge_fwd]
                    twisted["r"][edge_bwd] ^= x_loops["b"][edge_bwd]
                elif vertex_color == "b":
                    edge_bwd = edge_lookup[(node, -1)]
                    twisted["g"][edge_bwd] ^= x_loops["r"][edge_bwd]
                    if t_index > 0:
                        node2 = vertex_index(
                            (x_index - 1) % linear_size,
                            (y_index - 1) % linear_size,
                            t_index - 1,
                            linear_size,
                        )
                        edge_fwd2 = edge_lookup[(node2, 1)]
                        twisted["r"][edge_fwd2] ^= x_loops["g"][edge_fwd2]
                elif vertex_color == "r":
                    edge_fwd = edge_lookup[(node, 1)]
                    twisted["g"][edge_fwd] ^= x_loops["b"][edge_fwd]
                    if t_index < time_depth - 1:
                        node2 = vertex_index(
                            (x_index + 1) % linear_size,
                            (y_index + 1) % linear_size,
                            t_index + 1,
                            linear_size,
                        )
                        edge_bwd2 = edge_lookup[(node2, -1)]
                        twisted["b"][edge_bwd2] ^= x_loops["g"][edge_bwd2]
    return twisted


def twisted_delegated_accountant(
    context: MultiLayerContext,
    time_step: int,
    parent: ParentLayerView,
) -> ColorConfig:
    """Herald the links twisted by the loops the parent decoder added this step.

    Causal analogue of the Completing-the-Loop heralding (Sec. IV of
    arxiv/2604.02033): there the heralded links come from the delta between
    the JIT and global corrections, which is only available after the fact.
    Just in time, the decoder-visible proxy is the loop configuration the
    parent's own correction added in the current step
    (correction ^ previous_correction), pushed through the deterministic
    full-lattice twist rules.
    """
    correction_delta = {
        color: (parent.correction[color] ^ parent.previous_correction[color]).astype(np.uint8)
        for color in COLORS
    }
    return generate_twisted_z_errors(
        correction_delta,
        context.linear_size,
        context.time_depth,
        context.edge_lookup,
        is_full=True,
        twist_masks=context.twist_masks,
    )


def make_default_layer_specs(probabilities: Sequence[float]) -> List[LayerSpec]:
    """Default N-layer stack: X-type base layer, twisted-Z-type layers above it."""
    specs = [LayerSpec(noise_probability=probabilities[0], error_type="x")]
    for probability in probabilities[1:]:
        specs.append(
            LayerSpec(
                noise_probability=probability,
                error_type="z",
                generate_delegated_errors=twisted_delegated_generator,
                account_delegated_errors=twisted_delegated_accountant,
            )
        )
    return specs


def jit_decode_step_weighted(
    context: MultiLayerContext,
    time_step: int,
    noise: np.ndarray,
    current_prediction: np.ndarray,
    heralded_links: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One JIT step (prefix decode + full rejoin) with optional heralded links.

    Mirrors decoder.jit_decode_step; when heralded_links is a nonzero mask,
    both the prefix and full matchings are rebuilt with weight 0 on the
    heralded links so the decoder can place correction there for free.
    """
    prefix_incidence = context.prefix_incidences[time_step]
    if heralded_links is not None and heralded_links.any():
        full_weights = np.ones(context.num_edges, dtype=np.float64)
        full_weights[heralded_links.astype(bool)] = 0
        full_matching = Matching(context.full_incidence, weights=full_weights)
        prefix_width = prefix_incidence.shape[1]
        prefix_weights = np.ones(prefix_width, dtype=np.float64)
        limit = min(prefix_width, context.num_edges)
        prefix_weights[:limit] = full_weights[:limit]
        prefix_matching = Matching(prefix_incidence, weights=prefix_weights)
    else:
        full_matching = context.full_matching
        prefix_matching = context.prefix_matchings[time_step]

    syndrome = prefix_incidence @ noise[: prefix_incidence.shape[1]] % 2
    if np.count_nonzero(syndrome) % 2 == 1:
        syndrome[-1] = 1
    step_prediction = prefix_matching.decode(syndrome)
    joined = current_prediction.copy()
    joined[: len(step_prediction)] += step_prediction
    joined_syndrome = context.full_incidence @ joined % 2
    return full_matching.decode(joined_syndrome)


def run_multilayer_jit(
    context: MultiLayerContext,
    layer_specs: Sequence[LayerSpec],
    base_noises: Sequence[ColorConfig],
):
    """Run the interleaved multi-layer JIT protocol on one noise realization.

    base_noises: per layer, {color: uint8 edge array} of intrinsic noise (the
    delegated contributions are generated inside this function).

    Returns (predictions, delegated, residuals), each a list over layers of
    per-color full-lattice arrays; residuals are mod-2 (noise ^ correction).
    """
    num_layers = len(layer_specs)
    predictions = [
        {color: np.zeros(context.num_edges, dtype=np.uint8) for color in COLORS}
        for _ in range(num_layers)
    ]
    previous_corrections = [
        {color: np.zeros(context.num_edges, dtype=np.uint8) for color in COLORS}
        for _ in range(num_layers)
    ]
    delegated = [
        {color: np.zeros(context.num_edges, dtype=np.uint8) for color in COLORS}
        for _ in range(num_layers)
    ]

    for time_step in range(context.time_depth):
        for layer_index, spec in enumerate(layer_specs):
            heralded = None
            if layer_index > 0:
                parent = layer_index - 1
                parent_noise = {
                    color: base_noises[parent][color] ^ delegated[parent][color]
                    for color in COLORS
                }
                parent_view = ParentLayerView(
                    correction={
                        color: (predictions[parent][color] % 2).astype(np.uint8)
                        for color in COLORS
                    },
                    previous_correction={
                        color: previous_corrections[parent][color] for color in COLORS
                    },
                    residual={
                        color: ((parent_noise[color] + predictions[parent][color]) % 2).astype(np.uint8)
                        for color in COLORS
                    },
                )
                if spec.generate_delegated_errors is not None:
                    new_delegated = spec.generate_delegated_errors(context, time_step, parent_view)
                    for color in COLORS:
                        delegated[layer_index][color] ^= new_delegated[color].astype(np.uint8)
                if spec.account_delegated_errors is not None:
                    heralded = spec.account_delegated_errors(
                        context, time_step, replace(parent_view, residual=None)
                    )

            layer_noise = {
                color: base_noises[layer_index][color] ^ delegated[layer_index][color]
                for color in COLORS
            }
            for color in COLORS:
                previous_corrections[layer_index][color] = (
                    predictions[layer_index][color] % 2
                ).astype(np.uint8)
                predictions[layer_index][color] = predictions[layer_index][color] + jit_decode_step_weighted(
                    context,
                    time_step,
                    layer_noise[color],
                    predictions[layer_index][color],
                    heralded[color] if heralded is not None else None,
                )

    residuals = [
        {
            color: (
                (base_noises[layer_index][color] ^ delegated[layer_index][color])
                + predictions[layer_index][color]
            )
            % 2
            for color in COLORS
        }
        for layer_index in range(num_layers)
    ]
    return predictions, delegated, residuals


def run_global_cascade(
    context: MultiLayerContext,
    layer_specs: Sequence[LayerSpec],
    base_noises: Sequence[ColorConfig],
):
    """Baseline: decode the layer stack globally (full 3D MWPM per layer).

    Delegated errors for layer i are generated from layer i-1's *global*
    residual by summing the layer's incremental generator over all time steps,
    so JIT and global cascades share the same generation model. Returns the
    per-layer residuals.
    """
    residuals: List[ColorConfig] = []
    parent_view: Optional[ParentLayerView] = None
    zeros = {color: np.zeros(context.num_edges, dtype=np.uint8) for color in COLORS}

    for layer_index, spec in enumerate(layer_specs):
        noise = {color: base_noises[layer_index][color].copy() for color in COLORS}
        if layer_index > 0 and spec.generate_delegated_errors is not None:
            for time_step in range(context.time_depth):
                new_delegated = spec.generate_delegated_errors(context, time_step, parent_view)
                for color in COLORS:
                    noise[color] ^= new_delegated[color].astype(np.uint8)

        correction = {}
        residual = {}
        for color in COLORS:
            syndrome = context.full_incidence @ noise[color] % 2
            correction[color] = context.full_matching.decode(syndrome)
            residual[color] = ((noise[color] + correction[color]) % 2).astype(np.uint8)
        residuals.append(residual)
        parent_view = ParentLayerView(
            correction=correction, previous_correction=zeros, residual=residual
        )
    return residuals


def layer_has_logical_error(
    context: MultiLayerContext,
    spec: LayerSpec,
    residual_by_color: ColorConfig,
) -> int:
    """Return 1 if any color of the layer's residual carries a logical error."""
    for color in COLORS:
        if is_logical_error(
            residual_by_color[color],
            context.linear_size,
            context.time_depth,
            error_type=spec.error_type,
            boundary=context.boundary,
            edge_endpoints=context.edge_endpoints,
        ):
            return 1
    return 0


def sample_base_noises(
    context: MultiLayerContext,
    layer_specs: Sequence[LayerSpec],
) -> List[ColorConfig]:
    """Sample the intrinsic (non-delegated) noise for every layer and color."""
    base_noises = []
    for spec in layer_specs:
        noise = {
            color: np.random.binomial(1, spec.noise_probability, context.num_edges).astype(np.uint8)
            for color in COLORS
        }
        for color in COLORS:
            noise[color][context.last_measured_edges] = 0
        base_noises.append(noise)
    return base_noises


def build_multilayer_filename(
    output_dir: str,
    boundary: str,
    probabilities: Sequence[float],
    linear_size: int,
    repetitions: int,
    run_id: int,
) -> str:
    """Canonical output filename for multi-layer runs (probabilities in layer order)."""
    probability_tag = "-".join(str(p) for p in probabilities)
    return (
        f"{output_dir}/MLJIT_{boundary}_p_{probability_tag}_L_{linear_size}"
        f"_reps_{repetitions}_{run_id}.pkl"
    )


def run_multilayer_simulation(
    linear_size: int,
    layer_specs: Sequence[LayerSpec],
    repetitions: int,
    output_dir: str = "results/multilayer",
    boundary: str = "OBC",
    run_id: int = 0,
    include_global_baseline: bool = True,
) -> dict:
    """Run the multi-layer JIT simulation and persist aggregate counters.

    Saved payload (dict) carries the run parameters plus, per decoding mode:
    - layer_errors[i]: repetitions where layer i's residual is a logical error;
    - cumulative_errors[i]: repetitions where any layer <= i failed (the
      multi-layer generalization of the two-layer combined counter);
    - any_error: repetitions where any layer failed (== cumulative_errors[-1]).
    """
    probabilities = [spec.noise_probability for spec in layer_specs]
    output_file = build_multilayer_filename(
        output_dir, boundary, probabilities, linear_size, repetitions, run_id
    )
    if os.path.exists(output_file):
        return pickle.load(open(output_file, "rb"))

    context = build_multilayer_context(linear_size, boundary)
    num_layers = len(layer_specs)
    jit_layer_errors = [0] * num_layers
    jit_cumulative_errors = [0] * num_layers
    global_layer_errors = [0] * num_layers
    global_cumulative_errors = [0] * num_layers

    for _rep in range(repetitions):
        base_noises = sample_base_noises(context, layer_specs)

        _, _, jit_residuals = run_multilayer_jit(context, layer_specs, base_noises)
        jit_flags = [
            layer_has_logical_error(context, spec, jit_residuals[layer_index])
            for layer_index, spec in enumerate(layer_specs)
        ]
        for layer_index in range(num_layers):
            jit_layer_errors[layer_index] += jit_flags[layer_index]
            jit_cumulative_errors[layer_index] += int(any(jit_flags[: layer_index + 1]))

        if include_global_baseline:
            global_residuals = run_global_cascade(context, layer_specs, base_noises)
            global_flags = [
                layer_has_logical_error(context, spec, global_residuals[layer_index])
                for layer_index, spec in enumerate(layer_specs)
            ]
            for layer_index in range(num_layers):
                global_layer_errors[layer_index] += global_flags[layer_index]
                global_cumulative_errors[layer_index] += int(any(global_flags[: layer_index + 1]))

    result = {
        "linear_size": linear_size,
        "boundary": boundary,
        "probabilities": probabilities,
        "error_types": [spec.error_type for spec in layer_specs],
        "repetitions": repetitions,
        "run_id": run_id,
        "include_global_baseline": include_global_baseline,
        "jit_layer_errors": jit_layer_errors,
        "jit_cumulative_errors": jit_cumulative_errors,
        "jit_any_error": jit_cumulative_errors[-1],
        "global_layer_errors": global_layer_errors,
        "global_cumulative_errors": global_cumulative_errors,
        "global_any_error": global_cumulative_errors[-1],
    }
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(result, open(output_file, "wb"))
    return result

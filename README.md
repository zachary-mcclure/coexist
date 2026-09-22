# Coexist

Differentiable alloy phase diagrams from production-engine energetics.

This repository accompanies the paper *Differentiable Alloy Phase Diagrams
from Production-Engine Energetics* (submitted to Computational Materials
Science). Common-tangent and invariant constructions are treated as implicit
layers, so that `jax.grad` through a fixed-iteration Newton solver returns
derivatives of liquidus, solidus, solvus, and invariant points with respect
to any upstream input of an unmodified production engine — engine-computed
energetics, sampled liquid interactions, a real interatomic-potential
parameter through an exact linear bridge, and a foundation model's own
internal composition coordinate. Where forward-only phase-diagram workflows
end at a diagram, differentiating the construction turns disagreement with
experiment into quantitative measurements of missing physics, and inverts to
fit engine parameters to experimental diagram features.

## Installation

Python ≥ 3.10 (developed on 3.11 in a `uv` venv):

```bash
uv venv --python 3.11
source .venv/bin/activate
pip install -e .
```

The two energetics engines the paper uses are optional extras:

```bash
pip install -e ".[mace]"     # MACE-MP-0 foundation-model potential
pip install -e ".[lammps]"   # LAMMPS eam/alloy coupling
```

## Tests

```bash
pytest tests/
```

The suite covers the implicit-layer solvers (common tangent, three-phase
invariant detection with the reaction classifier, ternary tie-triangles), the
free-energy corrections (quasichemical short-range order, Debye–Grüneisen
vibrations), the engine adapters (LAMMPS eam/alloy, MACE-MP-0), and the
finite-difference consistency of the differentiable boundaries.

## Reproducing the results

Every result regenerates from a single script in `examples/`. Each script
writes its numerical output as JSON under `examples/output/`, and the
figure-generation scripts render the figures. Interatomic potential files,
with NIST provenance notes, are under `potentials/`. The manuscript itself is
released with the publication (and its archived Zenodo record), not in this
code repository.

## Layout

- `coexist/` — the library: implicit-layer solvers (`core/`), free-energy
  models, and the production-engine adapters (`adapters/`, `layers/`)
- `examples/` — one script per result, plus figure generation
- `potentials/` — interatomic potential files with provenance
- `tests/` — pytest suite

## Citation

See `CITATION.cff`. The archived release of this repository carries a Zenodo
DOI, quoted in the paper's Data and code availability statement.

## License

Apache-2.0 — see `LICENSE`.

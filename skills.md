# Skills Guide

## Purpose

This file explains how to call the public skills in [skills.py](skills.py), and how to find the right skill quickly when the file grows too large to scan by eye.

The repository currently uses one large symbolic backend file, so the fastest workflow is usually:

1. Use `rg` or `grep` to find the candidate skill name.
2. Open the exact function in [skills.py](skills.py).
3. Call it with the correct signature.

There are now two redundant lookup layers:

1. Machine-readable lookup in `SKILL_REGISTRY` inside [skills.py](skills.py).
2. Human-readable lookup in [skill_registry.md](skill_registry.md) and this file.


## Public vs Private

- Public skills are the functions whose names do not start with `_`.
- Helper functions whose names start with `_` are internal utilities and should not be treated as stable external APIs.


## Calling Conventions

There are four public signature styles in [skills.py](skills.py).

### 1. Standard skill: `params -> dict`

This is the main pattern.

```python
from skills import lagrangian_equations
import sympy as sp

t = sp.Symbol("t", real=True, positive=True)
q = sp.Function("q")(t)
m, k = sp.symbols("m k", positive=True)

result = lagrangian_equations(
    {
        "T": sp.Rational(1, 2) * m * sp.diff(q, t) ** 2,
        "V": sp.Rational(1, 2) * k * q ** 2,
        "coords": [q],
        "time": t,
    }
)
```

Most physics skills follow this pattern and return a dictionary containing SymPy expressions, matrices, equations, or boolean checks.

### 2. Direct utility function: direct arguments

Some public utilities do not use a `params` dictionary.

```python
from skills import vector_divergence, commutator
import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
div_value = vector_divergence([x, y, z], coord_system="cartesian")

A = sp.Matrix([[0, 1], [1, 0]])
B = sp.Matrix([[1, 0], [0, -1]])
comm = commutator(A, B)
```

### 3. Zero-argument utility

```python
from skills import pauli_matrices

pauli = pauli_matrices()
```

### 4. `params -> Expr` special case

`thermodynamic_partial` is public, but it returns a SymPy expression rather than a dictionary.

```python
from skills import thermodynamic_partial
import sympy as sp

a, b = sp.symbols("a b")
expr = thermodynamic_partial(
    {
        "X": a + b,
        "Y": a - b,
        "Z": a * b,
        "a": a,
        "b": b,
    }
)
```


## Dynamic Dispatch

If the caller only knows the skill name as a string, use dynamic dispatch.

```python
import skills

skill_name = "partition_function"
skill = getattr(skills, skill_name)
result = skill(
    {
        "energies": [0, 1],
    }
)
```

This is the recommended pattern for agent-style or tool-style orchestration.


## Runtime Helpers

The codebase now provides real runtime helpers in [skills.py](skills.py):

- `get_skill_entry(skill_name)`
- `search_skills(query, module=None, call_style=None, limit=None)`
- `invoke_skill(skill_name, payload=None, include_trace=False)`

The preferred runtime entry point is now `invoke_skill`, not handwritten `getattr`.

```python
from skills import invoke_skill, search_skills

matches = search_skills("relativity", limit=3)
result = invoke_skill("partition_function", {"energies": [0, 1]})
```

Call-style notes for `invoke_skill`:

- `params_dict` and `params_expr` skills take a plain payload dict.
- `zero_arg` skills should be called with no payload.
- `direct_args` skills should be called as `{"args": [...], "kwargs": {...}}`.

Example for a direct-argument skill:

```python
from skills import invoke_skill

result = invoke_skill(
    "vector_divergence",
    {
        "args": [[1, 0, 0]],
        "kwargs": {"coord_system": "cartesian"},
    },
)
```


## Registry-First Dispatch

For safer orchestration, prefer `SKILL_REGISTRY` over raw `getattr` when the caller wants metadata plus a callable in one place.

```python
from skills import SKILL_REGISTRY

entry = SKILL_REGISTRY["partition_function"]
skill = entry["callable"]
result = skill({"energies": [0, 1]})
```

Useful registry fields:

- `callable`
- `module`
- `section`
- `call_style`
- `signature`
- `returns`
- `summary`
- `keywords`

If the caller only knows the physics topic and not the exact function name, start with [skill_registry.md](skill_registry.md), then open the exact function in [skills.py](skills.py).


## Hard-Rule Check Skill

The deterministic hard-rule gate is:

- `tot_hard_rule_check(params)`

Typical call:

```python
from skills import tot_hard_rule_check

result = tot_hard_rule_check(
    {
        "equations": ["E = T + V"],
        "known_vars": {"m": 1, "v": 2},
        "required_equation_patterns": ["E = T + V"],
        "required_known_vars": ["m"],
    }
)
```

Returned fields:

- `passed`
- `violations`
- `checked`

Main hard-rule families now supported:

- equation presence and equation-pattern rules
- known-variable presence rules
- model rules: `required_models`, `forbidden_models`, `required_model_patterns`, `forbidden_model_patterns`
- boundary-condition rules: `required_boundary_condition_keys`, `forbidden_boundary_condition_keys`, `required_boundary_condition_patterns`, `forbidden_boundary_condition_patterns`, `required_boundary_conditions`, `forbidden_boundary_conditions`
- dimensional-equality rules
- sign and finiteness rules


## Search First, Then Open the Function

Because the skill file is long, prefer `rg` for lookup.

If you want a topic-to-skill map before searching function bodies, read [skill_registry.md](skill_registry.md) first.

### List all public functions

```bash
rg '^def [a-zA-Z0-9_]+\(' skills.py
```

### Find one exact skill

```bash
rg '^def lagrangian_equations\(' skills.py
rg '^def tot_hard_rule_check\(' skills.py
```

### Find by topic keyword

```bash
rg 'lagrangian|hamiltonian|noether|rigid' skills.py
rg 'maxwell|poynting|em_wave|vector_' skills.py
rg 'schrodinger|pauli|angular_momentum|perturbation' skills.py
rg 'thermo|partition|statistical' skills.py
rg 'lorentz|relativistic|four_vector|velocity_addition' skills.py
rg 'slit|grating|optical|lens|jones|stokes|doppler|standing_wave' skills.py
rg 'continuity|bernoulli|navier|euler_fluid|reynolds|poiseuille|stokes_drag|sound|surface' skills.py
rg 'dimension|hard_rule|boundary' skills.py
```

### Jump by module heading

```bash
rg '^# .*Module|^# =+' skills.py
```

### If `rg` is unavailable

```bash
grep -n '^def ' skills.py
grep -n 'lorentz\|relativistic\|four_vector' skills.py
```


## Skill Index

This section is an index, not a full parameter reference. Use `rg` to jump to the exact function body.

### Module 1: Theoretical Mechanics

- `lagrangian_equations(params)`
- `hamiltonian_equations(params)`
- `inertia_tensor(params)`
- `euler_rigid_body_equations(params)`

### Module 2: Electrodynamics

- `vector_divergence(F, coord_system="cartesian", coords=None)`
- `vector_curl(F, coord_system="cartesian", coords=None)`
- `vector_gradient(phi, coord_system="cartesian", coords=None)`
- `scalar_laplacian(phi, coord_system="cartesian", coords=None)`
- `maxwell_equations_check(params)`
- `fields_from_potentials(params)`
- `poynting_vector(params)`
- `em_wave_dispersion(params)`

### Module 3: Quantum Mechanics

- `commutator(A, B, simplify_result=True)`
- `schrodinger_1d(params)`
- `pauli_matrices()`
- `pauli_algebra(params)`
- `angular_momentum_eigenstates(params)`
- `perturbation_first_order(params)`

### Module 4: Thermodynamics and Statistical Physics

- `thermodynamic_potentials(params)`
- `thermodynamic_partial(params)`
- `partition_function(params)`
- `statistical_distributions(params)`

### Module 5: Special Relativity

- `lorentz_boost_matrix(params)`
- `lorentz_transform_event(params)`
- `four_vector_inner_product(A, B)`
- `relativistic_energy_momentum(params)`
- `velocity_addition(params)`

### Module 6: Optics and Waves

- `multi_slit_intensity(params)`
- `grating_equation(params)`
- `single_slit_diffraction(params)`
- `ray_translation_matrix(d)`
- `ray_refraction_matrix(n1, n2, R=None)`
- `thin_lens_matrix(f)`
- `mirror_matrix(R)`
- `optical_system(params)`

### Module 7: Extended Utilities

- `noether_conservation(params)`
- `effective_potential_analysis(params)`
- `special_functions(params)`
- `error_propagation(params)`
- `dimensional_analysis(params)`
- `thick_lens(params)`
- `aberrations(params)`
- `jones_calculus(params)`
- `stokes_mueller(params)`
- `doppler_classical(params)`
- `standing_wave_modes(params)`
- `tot_hard_rule_check(params)`

### Module 8: Fluid Mechanics

- `continuity_equation(params)`
- `bernoulli_equation(params)`
- `euler_fluid_equation(params)`
- `navier_stokes_check(params)`
- `vorticity_and_stream(params)`
- `reynolds_number(params)`
- `poiseuille_flow(params)`
- `stokes_drag(params)`
- `sound_speed(params)`
- `surface_tension(params)`


## Recommended Usage Pattern for Large Callers

If the upper layer does not know the exact function name yet:

1. Search by topic with `rg`.
2. Select the exact skill name.
3. Import the function directly or load it with `getattr`.
4. Pass SymPy-friendly values in `params`.
5. Expect SymPy expressions or matrices in the result.

Minimal dispatcher example:

```python
import skills

def call_skill(skill_name: str, params: dict):
    skill = getattr(skills, skill_name)
    return skill(params)
```

Registry-based dispatcher example:

```python
from skills import SKILL_REGISTRY

def call_registered_skill(skill_name: str, params: dict):
    entry = SKILL_REGISTRY[skill_name]
    return entry["callable"](params)
```


## Practical Notes

- Most outputs are symbolic and may need `sp.simplify`, `sp.expand`, or `.subs(...)` downstream.
- Many functions accept either raw Python values or SymPy expressions because the implementation uses `sp.sympify` internally.
- If a skill raises `ValueError` or `NotImplementedError`, inspect the function body in [skills.py](skills.py) to see the exact currently supported mode.
- For long-term maintenance, keep adding new public skills under a clear module heading and keep this index updated.
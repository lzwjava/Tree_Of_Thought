# Skill Registry

## Purpose

This file is the human-readable companion to `SKILL_REGISTRY` in [skills.py](skills.py).

Use this file when you know the physics task or phenomenon but do not yet know the exact skill name.

Use [skills.md](skills.md) when you already know the skill name and need the calling convention.


## Redundant Lookup Paths

There are now two intentionally redundant lookup paths.

### 1. Machine-safe path

Use the runtime helpers in [skills.py](skills.py).

```python
from skills import get_skill_entry, invoke_skill, search_skills

matches = search_skills("maxwell", limit=3)
entry = get_skill_entry("maxwell_equations_check")
result = invoke_skill(
    "maxwell_equations_check",
    {
        "E": [0, 0, 0],
        "B": [0, 0, 0],
    }
)
```

If the caller already knows the exact skill name, `invoke_skill` is the safest single entry point.

### 2. Human-safe path

1. Start from this file to narrow the problem class.
2. Use `rg` or `grep` to jump to the exact function in [skills.py](skills.py).
3. Read [skills.md](skills.md) for the precise call pattern.


## Problem-to-Skill Map

### Analytical Mechanics

- Derive equations of motion from `T` and `V`: `lagrangian_equations`
- Convert a Lagrangian to Hamiltonian form: `hamiltonian_equations`
- Build a rigid-body inertia tensor: `inertia_tensor`
- Write body-frame rigid rotation equations: `euler_rigid_body_equations`
- Detect conserved quantities from symmetries: `noether_conservation`
- Analyze stability around an effective potential minimum: `effective_potential_analysis`

### Vector Calculus and Field Operators

- Compute divergence: `vector_divergence`
- Compute curl: `vector_curl`
- Compute a scalar gradient: `vector_gradient`
- Compute a scalar Laplacian: `scalar_laplacian`

### Maxwell and Electromagnetism

- Check whether a field configuration satisfies Maxwell equations: `maxwell_equations_check`
- Build `E` and `B` from scalar/vector potentials: `fields_from_potentials`
- Compute electromagnetic energy flux or density: `poynting_vector`
- Build vacuum, dielectric, or conductor dispersion relations: `em_wave_dispersion`

### Quantum Mechanics

- Evaluate an operator commutator: `commutator`
- Solve 1D stationary Schrodinger problems: `schrodinger_1d`
- Get Pauli matrices directly: `pauli_matrices`
- Manipulate Pauli combinations or spin eigenproblems: `pauli_algebra`
- Build spherical-harmonic angular-momentum eigenstates: `angular_momentum_eigenstates`
- Compute first-order non-degenerate perturbation corrections: `perturbation_first_order`

### Thermodynamics and Statistical Physics

- Derive thermodynamic potentials and Maxwell relations: `thermodynamic_potentials`
- Compute constrained thermodynamic partial derivatives: `thermodynamic_partial`
- Build the canonical partition function and derived observables: `partition_function`
- Return MB, FD, or BE distributions: `statistical_distributions`

### Special Relativity

- Build a Lorentz boost matrix: `lorentz_boost_matrix`
- Transform a spacetime event: `lorentz_transform_event`
- Compute a Minkowski inner product: `four_vector_inner_product`
- Solve relativistic energy-momentum relations: `relativistic_energy_momentum`
- Apply 1D relativistic velocity addition: `velocity_addition`

### Optics and Waves

- Compute multi-slit intensity distributions: `multi_slit_intensity`
- Solve the grating equation: `grating_equation`
- Compute single-slit diffraction: `single_slit_diffraction`
- Get ABCD free-space translation matrix: `ray_translation_matrix`
- Get ABCD refraction matrix: `ray_refraction_matrix`
- Get ABCD thin-lens matrix: `thin_lens_matrix`
- Get ABCD spherical-mirror matrix: `mirror_matrix`
- Compose a full paraxial optical system: `optical_system`
- Build a thick-lens effective model: `thick_lens`
- Estimate spherical/chromatic aberrations: `aberrations`

### Polarization and Wave Optics

- Work in Jones formalism: `jones_calculus`
- Work in Stokes/Mueller formalism: `stokes_mueller`
- Compute a classical Doppler shift: `doppler_classical`
- Enumerate standing-wave modes with boundary conditions: `standing_wave_modes`

### Fluids and Continuum Flow

- Check mass conservation: `continuity_equation`
- Build or solve Bernoulli relations: `bernoulli_equation`
- Check Euler-fluid residuals: `euler_fluid_equation`
- Check incompressible Navier-Stokes residuals: `navier_stokes_check`
- Convert stream function to flow field or compute vorticity: `vorticity_and_stream`
- Compute Reynolds number and regime hints: `reynolds_number`
- Compute Hagen-Poiseuille pipe flow quantities: `poiseuille_flow`
- Compute Stokes drag or settling speed: `stokes_drag`
- Compute sound speed in fluids or gases: `sound_speed`
- Compute Laplace pressure or capillary rise: `surface_tension`

### Validation, Scaling, and Utility Skills

- Evaluate uncertainty propagation: `error_propagation`
- Perform dimensional analysis and Buckingham Pi reduction: `dimensional_analysis`
- Build explicit special functions: `special_functions`
- Run deterministic hard-rule veto checks over equations, variables, models, dimensions, and boundary conditions: `tot_hard_rule_check`


## Search Recipes

When this file narrows the task but not yet to one exact function, jump with `rg`.

### Find all public skills

```bash
rg '^def [a-zA-Z0-9_]+\(' skills.py
```

### Mechanics and conservation

```bash
rg 'lagrangian|hamiltonian|inertia|rigid|noether|effective_potential' skills.py
```

### Electromagnetism and vector calculus

```bash
rg 'vector_|maxwell|potentials|poynting|em_wave' skills.py
```

### Quantum

```bash
rg 'commutator|schrodinger|pauli|angular_momentum|perturbation' skills.py
```

### Thermodynamics and statistics

```bash
rg 'thermodynamic|partition|statistical' skills.py
```

### Relativity

```bash
rg 'lorentz|four_vector|relativistic|velocity_addition' skills.py
```

### Optics and polarization

```bash
rg 'slit|grating|optical|lens|mirror|jones|stokes|doppler|standing_wave' skills.py
```

### Fluids

```bash
rg 'continuity|bernoulli|euler_fluid|navier|vorticity|reynolds|poiseuille|stokes_drag|sound|surface' skills.py
```

### Validation and hard rules

```bash
rg 'error_propagation|dimensional_analysis|tot_hard_rule_check' skills.py
```


## Registry-First Programmatic Pattern

If an upper-layer planner wants to search first and call later, use this pattern.

```python
from skills import SKILL_REGISTRY

def call_registered_skill(skill_name: str, params: dict):
    entry = SKILL_REGISTRY[skill_name]
    return entry["callable"](params)
```

Useful metadata fields inside each registry entry:

- `module`
- `section`
- `call_style`
- `signature`
- `returns`
- `summary`
- `keywords`


## When Unsure

Use this fallback order.

1. Start from this file by topic.
2. Confirm the exact function name in `SKILL_REGISTRY` or with `rg`.
3. Read [skills.md](skills.md) for call style and examples.
4. Open the exact implementation in [skills.py](skills.py) if parameter semantics still matter.
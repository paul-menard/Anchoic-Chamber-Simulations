## Repo summary

This is a small numerical PDE demo repository (finite difference / Helmholtz solver) intended for student exercises.
Primary entrypoint: `demo_control_polycopie.py` which wires together the three logical components:

- `preprocessing.py` — geometry, domain partitioning, right-hand sides and helper utilities (functions starting with `_set_...`, `set2zero`, fractal helpers).
- `processing.py` — matrix assembly and solver logic (stiffness/mass matrices, boundary condition assembly, `solve_helmholtz`).
- `postprocessing.py` — plotting helpers that write image files like `fig_u0_re.jpg`, `fig_un_re.jpg`, etc.
- `_env.py` — small constants for node types (NODE_INTERIOR, NODE_ROBIN, etc.).

## Big-picture architecture for an AI coding agent

- Data flow: `demo_control_polycopie.py` calls `preprocessing._set_*` to create domain, `processing.solve_helmholtz(...)` to compute the PDE solution, and `postprocessing._plot_*` to save visualizations. Modifications to behaviour are usually localized to one of those modules.
- Conventions: helper functions often begin with `_` (e.g. `_set_geometry_of_domain`, `_set_chi`) and are called from the main demo script. Public utility functions typically live at top-level (no package layout).
- Numerical shape assumptions: domain arrays are (M, N) and many functions flatten into K=M*N linear systems using index row = i*N + j. Be careful when changing indexing.

## Common tasks and how to run locally

- Create a Python environment and install the minimal dependencies: numpy, scipy, matplotlib.
- Run the demo in PowerShell from the repo root:

```powershell
python .\demo_control_polycopie.py
```

- The main demo uses parameters near the bottom of `demo_control_polycopie.py` (e.g., N, M, `wavenumber`, `Alpha`, `mu`, `V_obj`). Change those only in the top-level script unless you intend to alter global behaviour.

## Project-specific patterns worth preserving

- Boundary nodes encoded as integer constants in `_env.py`. Many algorithms branch on these values (use `preprocessing` helpers to set domain nodes rather than changing constants in multiple places).
- Matrix assembly is done with scipy.sparse.lil_matrix then passed to sparse solvers — avoid replacing this with dense arrays for moderate grid sizes.
- Plotting functions save images by default (e.g. `postprocessing._plot_uncontroled_solution`); they do not open interactive windows when running in scripts. Use those named functions for reproducible outputs.

## Integration points / places to modify

- If changing PDE coefficients or boundary setups, update `preprocessing._set_coefficients_of_pde` and `_set_rhs_of_pde` (these return arrays passed to `solve_helmholtz`).
- To change the solver behaviour, edit `processing.solve_helmholtz` or helper assembly functions such as `compute_stiffness_matrix`, `compute_mass_matrix`, and the `compute_*_condition` functions.
- Optimization/iterative routines should be added outside the core assembly functions (see commented hooks in `demo_control_polycopie.py`). Keep matrix assembly + solve separate from optimization loops.

## Examples (explicit references)

- To find where domain nodes are created: `preprocessing._set_geometry_of_domain(M, N, level)`
- Where the linear system is assembled and solved: `processing.solve_helmholtz(...)` (look for `compute_stiffness_matrix` etc.)
- Where images are written: `postprocessing._plot_uncontroled_solution(u0, chi0)` writes `fig_u0_re.jpg`/`fig_u0_im.jpg`.

## Gotchas discovered from code

- There is no requirements file or packaging; the agent should not assume extras. Use only numpy/scipy/matplotlib and mention missing dependencies to the user.
- Many functions rely on side-effecting numpy arrays and in-place mutation (e.g., `set2zero` mutates arrays). Prefer copying when prototyping changes.

## What to do when adding tests or refactors

- Add tests that run small grids (small N) so solves and plots are fast. Use `N=8` or `N=10` for unit tests.
- When refactoring, keep the three-layer separation (preprocess / processing / postprocess) and preserve the index flattening scheme `row = i*N + j` to avoid subtle bugs.

---

If anything here is unclear or you want more agent examples (unit tests, a CI job, or a suggested requirements.txt), tell me which part to expand or modify. I'll update this file accordingly.

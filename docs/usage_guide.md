# Usage guide: families and single models

A short, practical guide to running `rotating_bs_solver`. The full parameter list is in the
[README](../README.md#parameter-reference), and `rotating_params.par` is a template with every
parameter and a one-line note on each.

## 1. Build and run

```bash
make                                   # builds ./rotating_bs_solver
mkdir -p out/myrun && cd out/myrun     # output goes to the CURRENT directory: one directory per job
../../rotating_bs_solver my.par > log  # my.par: the parameter file
```

- The parameter file is plain text, one `name = value` per line; `#` starts a comment.
- Parameters you leave out take the defaults in `bs_params.h`.
- Seed or restart paths inside the file (`seed_file`, `init_file`) are relative to where you run.
- Units: $G = 1$, $\mu = 1$ (boson mass). `kappa_target = 25.1327` is $8\pi$, the usual $G=1$ coupling.

## 2. What you choose

| choice | parameter | values |
|---|---|---|
| spherical or rotating | `sequence_mode` | `spherical` (1D) / `rotating` (2D) |
| matter | `matter` | `scalar` (uses `potential`), `proca`, `dirac` |
| potential (scalar) | `potential` | `mini`, `solitonic` (`sigma`), `kkls`, `axionic`, `repulsive`, `kleihaus` |
| azimuthal number | `m_azimuthal` | 1, 2, … (Dirac: 1 means $m=\tfrac12$) |
| grid | `Nr`, `Ntheta`, `radial_grid` | e.g. 150×41, `adaptive` for strong gravity |
| starting model | `omega_target`, seed | see section 3 |

## 3. How a run starts

The solver needs a first model, at `omega_target`, built from an analytic seed.

- **Potentials with Q-balls** (`solitonic`, `kkls`, `axionic`, `kleihaus`): a rotating Q-ball is
  solved at $\kappa=0$ from a torus seed (`rot_seed_amp`, `rot_seed_width`), then the gravitational
  coupling is ramped to `kappa_target` in steps of `delta_kappa`.
- **`mini`, `repulsive`, Proca, Dirac:** no flat-space solution, so the first model is solved
  directly at `kappa_target`.
  - mini and repulsive: the torus seed works, e.g. `omega_target = 0.9`, `rot_seed_amp = 0.02`,
    `rot_seed_width = 6`.
  - Proca and Dirac: give a radial seed with `seed_file` (see `examples/seeds/`).
- **Restart:** `init_file = bs2d.dat` starts from a profile saved by an earlier run (any grid).
  Adding `init_file_prev = bs2d_prev.dat` restarts right at a fold.

If the first solve fails ("Initial solve ... failed"), the seed is the usual cause: try a smaller
`rot_seed_amp`, a different `rot_seed_width`, or an `omega_target` closer to the mass (0.9–0.95).

## 4. Generating a family

A family is the whole curve of models, for example $M(\omega)$, with pseudo-arclength continuation.
It follows the curve through turning points in $\omega$, $Q$ and $M$.

```
sequence_mode = rotating
continuation_mode = arclength
potential = mini
m_azimuthal = 1
Nr = 200
Ntheta = 41
radial_grid = adaptive
kappa_target = 25.132741228718345
omega_target = 0.9
rot_seed_amp = 0.02
rot_seed_width = 6
delta_omega = 0.01        # sign = direction: positive toward lower omega, negative toward omega -> 1
arclength_max_steps = 300
first_law_tol = 0.01
light_rings = 1           # optional diagnostics
ergoregion = 1
```

- **Direction.** Run it twice, with `delta_omega = 0.01` and `-0.01`, to cover both sides of the
  starting model. Use separate directories.
- **Where it stops:** `arclength_max_steps`, `arclength_omega_min` / `arclength_omega_max`, or a
  guard (profile under-resolved, first law $dM=\omega\,dQ$ violated). A guard stop usually means
  "raise `Nr`/`Ntheta`", not "loosen the guard".
- **Output:**
  - `families2d.dat`, one line per model: $\kappa$, $\omega$, `phi_max`, $M$, $Q$, $J$, … (columns in
    the README's output-files section);
  - `bs2d.dat`, the latest profile.
- **Accuracy check.** Compare `mass_komar` with `mass_asym`, and `angmom` with $m\cdot$`charge`.
- **Continue or refine later.** Restart with `init_file = bs2d.dat`, `init_file_prev = bs2d_prev.dat`
  on a finer grid and a small `delta_omega`.

Plot it:

```python
import numpy as np, matplotlib.pyplot as plt
a = np.loadtxt("families2d.dat")
plt.plot(a[:, 1], a[:, 3]); plt.xlabel("omega"); plt.ylabel("M"); plt.show()
```

## 5. Generating a single model

Add one line to the same parameter file:

```
single_model_phi_max = 0.1     # peak amplitude of the model you want
```

The solver then follows the family from `omega_target` (direction from the sign of `delta_omega`)
until `phi_max` is bracketed, refines to $10^{-9}$ relative, writes the profile to **`bs2d.dat`**
and appends the model to `families2d.dat`. It works through turning points, so the target can lie
past the first fold.

- `phi_max` is the **maximum** of the field amplitude. For $m\ge1$ the field is zero on the axis,
  so a "central" amplitude would be 0.
- Check that the target is reachable: `phi_max` along the family is in column 3 of
  `families2d.dat`. If it is never reached, the run ends at `arclength_max_steps` or at a guard, and
  no profile is written for the target.
- The log ends with a line `single model: omega = ... phi_max = ... M = ... Q = ... J = ...`.
- To get the model at a given **$\omega$** instead, set `omega_target` to it, `continuation_mode = omega`
  and `delta_omega = 0`: the run stops after the first solve (and the $\kappa$ ramp) and writes
  `bs2d.dat`. This only converges for $\omega$ reachable from the seed (for mini, roughly
  0.8–0.98); otherwise use `single_model_phi_max`. A fixed-$Q$ model (scalar only) is
  `charge_target > 0` with a non-arclength mode.

Example (rotating mini boson star, $m=1$, $\kappa=8\pi$):

```
sequence_mode = rotating
potential = mini
matter = scalar
m_azimuthal = 1
Nr = 100
Ntheta = 21
kappa_target = 25.132741228718345
omega_target = 0.9
rot_seed_amp = 0.02
rot_seed_width = 6
delta_omega = 0.01
single_model_phi_max = 0.1
```

This gives $\omega=0.648$, $M=0.991$, $Q=0.897$ in about 5 s on a 100×21 grid. For production
numbers use a finer grid (e.g. 200×41 with `radial_grid = adaptive`) and compare two resolutions.

## 6. Reading a profile

`bs2d.dat` has columns `i j r theta f l g W <fields>`; the header records $\omega$, $\kappa$, $m$ and
the radial map. $\phi$ is the last column for scalar matter. The metric functions $f, l, g, W$ are
those of the Kleihaus–Kunz ansatz (see the README's conventions section).

## 7. Practical tips

- Start with a coarse grid (100×21) to see whether the setup works, then refine.
- One directory per run; redirect the log (`> log`) and look for `[warn]` lines.
- Strong-gravity models need `radial_grid = adaptive` and more points.
- Memory is dominated by the sparse factorization; see the README's Practical tips for the
  `linear_preconditioner` / `ilut_*` options.
- Ready-to-run examples are in `examples/` (README Quick start).

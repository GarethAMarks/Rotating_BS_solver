# Development notes (validation log)

The running log kept while the solver was developed: derivations, design decisions, every validation run and benchmark. Paths such as `runs/...` refer to run directories from the development working tree, which may not be included in the repository. The user-facing documentation is the top-level `README.md`.

---

# Rotating BS Solver (development log)

This repository now includes a **compile-ready residual solver prototype** for constructing boson-star solutions, starting with the spherical ($m=0$) baseline described in Appendix A of Siemonsen & East (2021) and Appendix B.1 of Kleihaus et al. (Ref. [62] in Siemonsen & East).

## Added scaffold components

- `main.cpp`: executable entrypoint
- `bs_params.h`: run configuration data model
- `params_io.h/.cpp`: parameter file reader (using `fill_parameter` from `mathutils`)
- `solver.h/.cpp`: spherical residual assembly + Newton-Krylov nonlinear solve (matrix-free Jacobian and restarted GMRES)
- `rotating_params.par`: parameter template for spherical + rotating roadmap
- `Makefile`: basic build and run targets

## Implemented equations and boundary conditions

The code currently:

1. Reads `rotating_params.par`
2. Builds a compactified radial grid using $\bar r \in [0,1]$
3. Initializes regular seed profiles for `{phi, metric_f, metric_l, omega_s, rho}`
4. Solves (iteratively) the **spherical** residual system:
	 - metric + scalar equations from Kleihaus et al. Appendix B.1, Eqs. (59)-(61)
	 - auxiliary equations from Appendix B.1, Eqs. (62)-(63)
	 - BCs from Eqs. (28)-(30) plus Appendix A auxiliary BCs (`omega` or `Q` continuation)

### Newton solver details

- Nonlinear method: damped Newton with Armijo backtracking line search
- Linear solve (`linear_solver`):
  - `banded` (default): finite-difference Jacobian assembled by coloring. Every residual at grid
    point $k$ depends only on fields at $k-2..k+2$, so perturbing every 5th point of one field
    at once recovers whole columns ($5\times n_{\rm fields}$ residual evaluations instead of
    $n_{\rm fields} N$). Fields are interleaved per grid point and solved with a banded LU.
    Couplings outside the band are added exactly as rank-1 (Woodbury) updates: the lapse-mode
    $\omega_s$ BC uses $f(0)$, and the arclength constraint row is dense (its derivative is
    exactly the tangent $t$).
    Central differences with step $\epsilon_{\rm mach}^{1/3}\max(|x_j|, 10^{-3}\,{\rm typ}_a)$,
    ${\rm typ}_a$ = max of field $a$. Forward differences (or an absolute step floor, which
    exceeds $f(0)\sim10^{-5}$ in compact stars) are not accurate enough for this
    ill-conditioned system and make Newton stall.
  - `direct`: dense FD Jacobian + dense LU. $O(N^3)$; kept as a cross-check (agrees with
    `banded` to ~10 digits).
  - `gmres`: matrix-free, unpreconditioned restarted GMRES.
- Convergence is tested componentwise: $|R_i| \le$ `rel_tol_residual` $\cdot s_i +$ `abs_tol_residual`
  for every row, where $s_i$ is the sum of absolute values of the terms entering $R_i$ (absolute
  FD stencils for derivatives). Round-off in $R_i$ is $\sim\epsilon_{\rm mach} s_i$, which grows
  like $1/h^2$ with resolution and like $1/f^2$ in the cores of compact stars, so a fixed
  absolute tolerance is unreachable there. A Newton stall (line-search failure, vanishing
  update) is accepted only if the same test holds with `stall_rel_tol`.
- A converged state with $\max|\phi| <$ `trivial_phi_tol` is rejected: $\phi=0$ with flat
  metric solves the system for every $\omega$.
- In `continuation_mode = omega`, the frequency field is treated as fixed and excluded from the Newton state vector.

### Continuation along a family (`continuation_mode = arclength`, default)

Pseudo-arclength continuation in the full discretized state $x$ (all fields, $\omega_s$ included).
From the model at `omega_target` one ordinary step of `delta_omega` gives a second point; then,
with $x_{\rm prev}, x_{\rm cur}$ the last two solutions,

$$
t = \frac{x_{\rm cur}-x_{\rm prev}}{|x_{\rm cur}-x_{\rm prev}|},\qquad
x_{\rm pred} = x_{\rm cur} + \Delta s\, t,\qquad
R(x)=0,\quad \langle t,\, x - x_{\rm pred}\rangle = 0 ,
$$

starting Newton from the extrapolated seed $x_{\rm pred}$. The constraint replaces the
$\omega_s$ boundary condition at infinity, so $\omega$ is an unknown. Each new model lies a
distance $\Delta s$ *ahead* of the last along $t$: turning points in $\omega$, $Q$ or $M$ need no
mode switching and the branch cannot be retraced (the old `omega` -> `charge` switching retraces
at the first $Q$ turning point it cannot pass, indefinitely). Steps whose secant turns by more than
`arclength_min_cos` are rejected; $\Delta s$ grows x1.5 when Newton needs <= 3 iterations,
shrinks x0.7 at >= 7 and halves on failure.

Continuation stops when the profile is no longer resolved, because the discrete branch then folds
back spuriously (the fold moves with `Nr`) and the continuation would follow it faithfully:
- `core_resolution_tol`: deep in the spiral the core shrinks onto $r=0$ in isotropic coordinates
  and $f$ varies across the first cell;
- `wall_resolution_tol`: thin-wall stars have a wall only a cell or two wide.
- `arclength_omega_max` (0: $m_B$): stop once $\omega$ rises to this value. Without it, an ascending leg
  creeps toward $\omega=1$ with $M\to0$ and the star growing without bound.
- `first_law_tol` (rotating solver only): consecutive models must satisfy $dM=\omega\,dQ$,
  measured as $|\Delta M-\bar\omega\Delta Q|/(|\Delta M|+\bar\omega|\Delta Q|+\bar Q|\Delta\omega|)$
  ($10^{-4}$ on resolved stretches). It catches spurious 2D branches that the profile indicators
  miss. For example, solitonic $\sigma=0.05$ at $400\times61$ shows fake "spirals" below $\omega\approx0.17$
  whose $\omega$ and $Q$ turning points coincide (local violation 5–95%). Grid noise, from a thin wall
  crossing cells, reaches a few percent, so the default is 0.05.
  On a violation the step is first halved up to `first_law_retries` (2) times; this can only
  postpone a stop, since on a sharply curving branch the midpoint rule's own error grows with the step.

Tested (vs independent shooting, see the 2026-10-29 notes): mini stars through
$\omega_{\min}$, $Q_{\min}$, $\omega_{\max}$, $Q_{\max}$ and the second $\omega_{\min}$ (`Nr = 1600`,
$M$ to ~1e-9); solitonic $\sigma=0.0424$ through $Q_{\max}\approx66$, $M_{\max}\approx10.3$ and four $\omega$
turning points (`Nr = 1600`, 40 s); KKLS through three $\omega$ turning points. At `Nr = 400` the
solitonic family stops at $\omega\approx0.23$ (wall resolution); use `Nr = 1600` for compact stars.

### Q-ball bootstrap + kappa continuation

- If `use_qball_bootstrap = 1`, the solver first attempts a flat-space (`kappa=0`) Q-ball shooting solve for Eq. (39), then maps that profile to the compactified grid.
- After that, it marches in `kappa` using `delta_kappa`, solving each stage with Newton-Krylov and using each converged stage as the next initial guess.
- Q-ball bootstrap is currently enabled only for potentials with attractive nonlinear terms: `kkls`, `solitonic`, `axionic`.
- For `mini` and `repulsive` there is no flat-space solution to continue from (at $\kappa=0$ the
  scalar equation is linear and only $\phi=0$ solves it), so the code skips the $\kappa$ ramp and
  solves directly at `kappa_target` from an analytic seed fitted to mini boson stars
  ($\phi(0)\approx(0.14+0.2\varepsilon^2)\varepsilon^2$, $1-f(0)\approx1.94\varepsilon^2$ with
  $\varepsilon^2 = 1-\omega^2/\mu^2$). This converges for $\omega/\mu \in [0.80, 0.98]$; start there and march in $\omega$.
- Whenever Q-ball shooting succeeds, the solver applies the exponential tail fit directly to the solved profile data (used for both seeding and output), then writes `qball.dat` with columns:
	- `r`, `phi`, `dphi`
	- `U` (potential density)
	- `grad_energy = (dphi)^2`
	- `time_energy = omega^2 phi^2`
	- `energy_density = grad_energy + time_energy + U`
	- `j_t = 2 omega phi^2` (Noether density for the flat-space shooting ansatz)
	- `shell_mass_density = 4 pi r^2 energy_density`
	- `shell_charge_density = 8 pi omega r^2 phi^2`
		- `phi_fit`, `dphi_fit` from the asymptotic model $A e^{-\beta r}/r$ (diagnostic fit columns)
	- Shooting now prefers the fundamental no-node branch by treating zero-crossing profiles as overshoots during bracketing/bisection.

### Potentials

`potentials.{h,cpp}` (shared): `mini`, `repulsive`, `solitonic`, `kkls`, `axionic` (Siemonsen & East
Eqs. 2-4) and `kleihaus`, $U=\lambda(\phi^6-a\phi^4+b\phi^2)$ (Kleihaus et al. Eq. 2; parameters
`kleihaus_lambda`, `kleihaus_a`, `kleihaus_b`, boson mass $\sqrt{\lambda b}$). `kappa_target = 0`
traces flat-space Q-ball families.

### Global quantities

- Mass: outside the star the metric is isotropic Schwarzschild, so
  $MG = 2r(1-\sqrt f)/(1+\sqrt f)$ evaluated at $r_{N-2}$ is exact (no $O(M/r)$ error); reported
  in units $G=\kappa/8\pi$ (so $G=1$ for $\kappa=8\pi$, and Kleihaus' tables at other $\kappa$ compare
  directly). At $\kappa=0$ the Komar mass is used.
- Komar mass $M_K = 4\pi\int(4\omega^2\phi^2/f-2U)\,l^{3/2}r^2/f\,dr$ (column `mass_komar`): agrees with
  the asymptotic $M$ to ~1e-8 (`Nr = 400`); equals the Q-ball energy in flat space.
- Charge: $Q = 8\pi r^2 \omega_s \partial_r\rho$ at $r_{N-2}$ (second-order accurate in $h$), and the
  volume integral $Q = 8\pi\omega\int l^{3/2}r^2\phi^2/f^2\,dr$ (column `charge_integral`, more accurate).
- Kleihaus et al. benchmarks reproduced: spherical cusps at $\kappa=0.4$ and $1$ ($Q$, $M$ to 5e-5 - 4e-4);
  flat Q-ball at $Q=410$: $\omega=0.59331$, $M=294.09$ (independent shooting: identical to 1e-5; the
  paper's 0.5927 / 293.8 differ by 1e-3, consistent with its stated precision).
- Validated against an independent areal-coordinate shooting code: for mini stars
  $M$ agrees to $\sim10^{-6}$ and $Q$ to $\sim10^{-5}$ at `Nr = 400` (e.g. $M_{\max}=0.63289$ at $\omega=0.85$).

### Notes on correction handling

- Siemonsen & East explicitly list corrections to the **axially symmetric** equations in Kleihaus et al. (their Appendix A, Eqs. (20)-(21)).
- The present implementation is intentionally the spherical baseline (B.1), which is the recommended seed stage before the full rotating 2D system.
- For `continuation_mode = omega`, `omega_s` is enforced as constant at the target value.
- For `continuation_mode = charge`, the asymptotic charge condition
	$$
	8\pi r^2\,\omega_s\,\partial_r\rho\big|_{\infty}=Q
	$$
	is enforced in the boundary residual.

## Rotating stars (`sequence_mode = rotating`)

`rotating_solver.{h,cpp}`: stationary axisymmetric stars with azimuthal number `m_azimuthal` on an
`Nr` x `Ntheta` grid, $\bar r\in[0,1]$ x $\theta\in[0,\pi/2]$ (equatorial symmetry).

- Metric $ds^2=-f\,dt^2+\frac{l}{f}[g(dr^2+r^2d\theta^2)+r^2\sin^2\theta(d\varphi-\frac{W}{r}dt)^2]$,
  $\Phi=\phi(r,\theta)e^{i(\omega t+m\varphi)}$. Unknowns $f,l,g,W,\phi$ per node plus the scalar $\omega$,
  closed by one constraint row: fixed $\omega$, fixed $Q$ (`charge_target`), or pseudo-arclength.
- Field equations: Kleihaus et al. B.2 + Siemonsen & East corrections, **generated** by
  `tools/derive_rotating.py` (sympy) into `rotating_equations.inc` / `rotating_densities.inc`. The
  script derives the Einstein-Klein-Gordon system from the metric and proves (exact rational
  arithmetic) that the generated equations follow from it; the uncorrected f-equation fails (negative
  control); the $m=0$ reduction equals the 1D equations. Re-run it after any change to the equations.
- Radial grid: uniform in $\xi\in[0,1]$ with an analytic map $\xi(r)$; radial derivatives via
  $\partial_r=\xi'\partial_\xi$, $\partial_r^2=\xi'^2\partial_\xi^2+\xi''\partial_\xi$, quadrature $dr=d\xi/\xi'$.
  The map is $\xi=\big(\frac{r}{L+r}+\sum_k\beta_kB_k(r)/B_k(\infty)\big)/(1+\sum_k\beta_k)$. Each
  cluster $B_k'$ is a top hat on $[r_a,r_b]$ with $\tanh$ edges of width $\delta$ and holds the fraction
  $\beta_k/(1+\sum\beta)$ of the points.
  - `radial_grid = compactified` (default): no clusters, $L$ = `radial_scale`. $L=1$ is the original
    $\bar r=r/(1+r)$, reproduced to $10^{-12}$.
  - `radial_grid = adaptive`: the map follows the solution. On every $\theta$ ray it finds the outer
    wall of $|\phi|$ and the inner rise (the first
    half-peak crossing). A **wall cluster** covers the outer walls of all rays, and a **core cluster**,
    symmetric about $r=0$, covers the inner rise. That rise becomes a spike in strongly relativistic
    cores, which reach $f(0)\sim10^{-4}$.
    - The outer wall is the steepest descent in $\ln r$ beyond the ray's peak, the maximum of
      $-r\,\partial_r|\phi|$, with thickness $2|\phi|/|\partial_r\phi|$ there. A half-peak criterion
      fails deep in the spiral: once the central spike exceeds twice the plateau, its own flank is
      taken for the wall and the map jumps.
  - `radial_cluster_fraction` (0.4) of the points go to the clusters. They are shared in proportion
    to how many thicknesses each spans, with at least 16 points each.
  - $L$ = `radial_scale_ratio` (0.5) × the outer edge of the wall cluster.
  - The map is refitted after the first solve, after every $\kappa$-ramp and $\omega$-march step, and
    after every arclength step. When it has drifted by more than `regrid_tol` (0.1, relative to the
    cluster width or $L$), the solver interpolates the states (6-point Lagrange in the old $\xi$) and
    re-converges them. In arclength mode both secant ends are re-converged, each with the arclength
    constraint through its interpolated state, which stays well posed at folds. If a re-solve fails,
    the old grid is kept.
  - The first-law residual on adaptive grids is $10^{-4}$–$10^{-3}$ on resolved stretches, so
    `first_law_tol = 0.01` is the useful setting there.
- **Ergoregion** (`ergoregion = 1`): the column `ergo_gtt_max` holds the largest
  $g_{tt}=-f+l\sin^2\theta\,W^2/f$ on the grid (after `light_rings`, before Proca's `lorenz`). A
  positive value means $\partial_t$ is spacelike somewhere, i.e. there is an ergoregion. For models
  that have one, the log gives the equatorial ergosurface crossings as circumferential radii
  $R=r\sqrt{l/f}$ in units of $M$; a torus crossing the equator has an inner and an outer radius.
- **Restart** (`init_file = path/bs2d.dat`): starts from a profile written by an earlier run, on any
  grid, with the same matter and $m$. The fields are interpolated onto the new grid with 4-point
  Lagrange in $r$ and $\theta$.
  - **Adaptive grids** reuse the radial map stored in the profile's header, so with the same `Nr` the
    nodes coincide. Without a stored map, the map is fitted to the profile in two passes. Refitting
    near a very compact core costs enough accuracy that Newton can fail.
  - **Continuation.** The profile's $\omega$ replaces `omega_target`. The arclength starts with a
    fixed-$\omega$ step **against** the sign of `delta_omega`, back where the family is single valued,
    and then continues forward. A forward step next to a fold can fail or leave the branch. Keep
    `delta_omega` small near folds; 0.0002 worked at the end of the Proca family.
  - With `write_bsdata_per_step = 1` (the default), `bs2d.dat` is rewritten after every accepted
    model, so a failed run leaves its last good model as a restart point.
  - Use a restart to carry the end of a family onto a finer grid without re-tracing it.
- **Light rings** (`light_rings = 1`): every model's equatorial light rings (circular photon orbits)
  are counted, and the count goes into an extra `light_rings` column of `families2d.dat`. The log
  lists each ring's circumferential radius $R=r\sqrt{l/f}$ in units of $M$, as co- or
  counter-rotating with the star.
  - **Method.** On the equator $g_{t\varphi}^2-g_{tt}g_{\varphi\varphi}=l r^2$, so the angular velocities
    of tangential photons are $H_\pm=W/r\pm f/(r\sqrt l)$. Light rings are their critical points
    (Cunha, Berti & Herdeiro 2017), found as sign changes of $\Delta H$ along the grid and refined by
    a parabola. The frame-dragging sign $W/r$ decides co- versus counter-rotating.
  - **Limitations.** Only the equatorial plane is searched. Off-equatorial rings are possible in
    principle but not expected for these stars.
  - **Tests** (`runs/maptest/lrtest.cpp`):
    - Schwarzschild in isotropic coordinates gives $R=3.000000\,M$.
    - The Lense–Thirring exterior gives $3M\mp(2/\sqrt3)a$ to $10^{-6}$, the first-order Kerr photon
      orbits, with the correct labels for both senses of rotation.
- Discretization: 1D $\xi$ stencils, 5-point $\theta$ stencils with parity ghosts (axis: $\phi$ parity
  $(-1)^m$; equator: even). Boundary rows (Kleihaus Eqs. 28-32): origin $\partial_rf=\partial_rl=0$,
  $g=1$, $W=0$, $\phi=0$; infinity $f=l=g=1$, $W=\phi=0$; axis $\partial_\theta f=\partial_\theta l=\partial_\theta W=0$, $g=1$, $\phi=0$.
- Newton: colored central-difference Jacobian (25 colors x 5 fields; checked against a dense
  Jacobian to 1e-13 relative with `BS_CHECK_JACOBIAN=1` on small grids), sparse LU (vendored Eigen,
  `third_party/eigen`), componentwise convergence test as in 1D.
- Route: analytic torus seed at $\kappa=0$ (`rot_seed_amp`, `rot_seed_width`) -> $\kappa$ ramp
  (`delta_kappa`) -> $\omega$ march (`delta_omega`, stops at `charge_target` crossing) or arclength
  (`continuation_mode = arclength`; negative `delta_omega` starts upward), with the 1D resolution guards.
- Output: `families2d.dat` (kappa omega phi_max mass_komar mass_asym charge angmom angmom_asym
  residual its), `bs2d.dat` (i j r theta f l g W phi). $M$, $Q$, $J$ by quadrature of the generated
  densities; `mass_asym`, `angmom_asym` from $f$, $W$ at the outer radii ($\kappa>0$).
- Sign convention: with $\omega,m>0$ the pattern rotates toward $-\varphi$, frames are dragged that
  way ($W<0$) and Kleihaus' $J=-\int T^t{}_\varphi\sqrt{-g}>0$, so $W\to-2JG/r^2$ (their Eq. 17 has $+$).
- Validation: $m=0$ equals the 1D solver (flat and gravitating; to 1e-10 in $\phi$, 4th-order
  $\theta$-quadrature convergence); Kleihaus Table 2 rotating Q-balls at $Q=410$: $m=1$
  $\omega=0.69339$, $M=363.52$ (paper 0.6931, 363.4), $m=2$ $\omega=0.7958$, $M=415.26$ (paper 0.7940, 414.7),
  converged in resolution (100x21 vs 200x41 differ by 1e-4) -- the paper's values carry ~1e-3 errors, as
  for its $n=0$ entry, which an independent shooting code confirms; $J=mQ$; $|J_{\rm asym}/J|=1$ to
  the fit accuracy; first law $dM=\omega\,dQ$ to ~1e-5 along families.
- Gravitating rotating stars: Kleihaus' first $n=1$ cusps at $\kappa=0.1,0.2,0.3$ reproduced:
  $(Q,M,\omega)=(910.95, 516.49, 0.3395)$, $(236.99, 179.17, 0.4327)$, $(110.92, 99.23, 0.5095)$ vs the
  paper's $(910.9, 516.5, 0.340)$, $(237.2, 179.3, 0.4325)$, $(111.0, 99.3, 0.510)$ (the table's blocks are
  printed in the order 0.2, 0.3, 0.1). Komar vs asymptotic $M$: <= 2e-4 (100x21), 5e-5 (200x41).
  The second cusps lie on very compact backbending branches beyond the resolution guards at 200x41.
- Mini $m=1$ ($\kappa=8\pi$, started directly from the seed since there is no Q-ball):
  $M_{\max}=1.31547$ at $\omega=0.77259$ at both 100x21 and 200x41; literature 1.315 (Yoshida & Eriguchi;
  Grandclement, Some & Gourgoulhon 2014).
- Siemonsen & East (2021) m=1 stars at $\kappa=8\pi$ (their Table I; masses from the commented column
  of their tex source; route: rotating Q-ball at $\omega=0.9$ -> $\kappa$ ramp -> $\omega$ march with
  `rot_omega_stop` = target; repulsive: direct start at $\kappa=8\pi$):

  | model | grid | $\mu M$ | $J/M^2$ | $C^K$ | $C^Q$ | S&E $\mu M$, $J/M^2$, $C^K$, $C^Q$ |
  |---|---|---|---|---|---|---|
  | KKLS $\kappa=0.1$, $\omega=0.382$ | 100x21, 200x41 | 1.8555 | 0.913 | 0.336 | 0.294 | 1.86, 0.91, 0.333, 0.290 |
  | solitonic $\sigma=0.05$, $\omega=0.303$ | 200x41 / 400x61 | 1.7095 / 1.7098 | 1.360 / 1.363 | 0.174 | 0.144 | 1.71, 1.36, 0.169, 0.140 |
  | axionic $f=0.005$, $\omega=0.303$ | 200x41 / 400x61 / 600x41 | 0.6718 / 0.6666 / 0.6669 | 3.548 / 3.595 / 3.594 | 0.063 | 0.051 | 0.66, 3.59, 0.061, 0.049 |
  | repulsive $\lambda=500$, $\omega=0.897$ | 200x41 / 400x81 | 1.6193 | 0.642 | 0.083 | 0.082 | 1.62, 0.64, 0.077, 0.078 |

  $M$ and $J/M^2$ agree to the published digits once resolved (the axionic star, with structure on the
  scale $f$, needs >= 400x61; 800x81 exceeds this machine's 7 GB in the sparse LU). Compactness uses the circular radius containing 99% of $M$ / $Q$, as
  described by S&E; it agrees to 1% for the most compact (KKLS) model and is 3-9% high elsewhere
  (also for their Table II: solitonic $\sigma=0.1$ at $\omega=0.8, 0.4$: 0.089, 0.306 vs 0.08, 0.28;
  axionic $f=0.01$ at $\omega=0.8, 0.3$: 0.034, 0.196 vs 0.03, 0.19). No single radius definition
  (circular, proper equatorial distance, areal, coordinate) reproduces all of their values, so their
  compactnesses were presumably measured differently (e.g. on their evolution grid); $M$, $J$ are the
  definition-independent checks.
- Full solitonic $\sigma=0.05$, $m=1$, $\kappa=8\pi$ family (`runs/solitonic_sigma005/`: `setup.sh`,
  `run.sh`, `plot.py`). Arclength from the $\omega=0.9$ model in both directions, timed with the solver's
  own clock:

  | grid | $\omega$ range | models | time (down + up) | peak RSS | stop (down / up) |
  |---|---|---|---|---|---|
  | 200x41 | 0.273 – 0.9966 | 58 | 91 + 53 = 144 s | 0.45 GB | first law / wall |
  | 400x61 | 0.138 – 0.9996 | 103 | 622 + 443 = 1065 s | 1.95 GB | first law / wall |

  Features:
  - a Newtonian local maximum $M=0.3371$ at $\omega=0.9857$;
  - a minimum $M=0.1708$ at $\omega=0.880$;
  - then $M$ rises monotonically to 7.30 ($C^K=0.374$) at $\omega=0.138$.

  The two grids agree to $\le10^{-5}$ in $M$ and $J/M^2$ for $0.5\le\omega\le0.98$, and to $10^{-3}$ at
  $\omega=0.35$. Below $\omega\approx0.3$ the 200x41 profiles show grid noise, and at $\omega\approx0.26$
  they jump to a spurious branch.

  At 400x61 the first-law residual stays below $10^{-3}$ down to $\omega=0.25$ and below $10^{-2}$ down to
  $\omega=0.21$. Below that it is noisy at the 1–4% level, so treat $\omega<0.2$ as qualitative. Without
  the first-law guard, the 400x61 run continued into fake turning points at $\omega\approx0.13$–0.14.
  There $\omega$ and $Q$ turned together, so they are discretization artifacts, not physical cusps.
- The same family on the **adaptive radial grid** (`runs/adaptive/`: `compare.py`,
  `plot_compare.py`; `first_law_tol = 0.01`, `arclength_omega_max = 0.9996` on the up legs).
  Run time is from the solver's clock:

  | grid | time (down + up) | peak RSS | resolved range | $M$ at $\omega=0.303$ |
  |---|---|---|---|---|
  | uniform 200x41 | 91 + 53 = 144 s | 0.45 GB | $\omega$ 0.30–0.996, first branch only | 1.70953 |
  | uniform 400x61 | 622 + 443 = 1065 s | 1.95 GB | $\omega$ 0.21–0.9996, first branch only | 1.70984 |
  | adaptive 150x41 | 347 + 71 = 418 s | 0.57 GB | 0.9997 → $\omega_{\min}$ → second branch to $\omega\approx0.17$ | – |
  | adaptive 200x61 | 852 + 244 = 1096 s | 2.08 GB | same | 1.71014 (300x61: 1.71018) |

  - **Turning point.** The adaptive grids resolve it with first-law residual $\sim10^{-4}$:
    $\omega_{\min}=0.1298$, then $M_{\max}=7.634\pm0.001$ and the maximum compactness
    $C^K_{\max}=0.4162$. The family then turns into the second (unstable) branch.
  - **Agreement between adaptive grids.** 150x41 and 200x61 agree to $\le3\times10^{-4}$ in $M$,
    $J/M^2$ and $C^K$ from $\omega=0.9$ through the turning point and along the second branch to
    $\omega\approx0.165$.
  - **Diffuse end.** They also agree to $10^{-6}$ for $0.9<\omega<0.99$ and to $5\times10^{-4}$ at
    $\omega=0.9997$. There uniform 200x41 is 2% off and uniform 400x61 13% off.
  - **Accuracy at $\omega=0.303$.** Against the 300x61 adaptive value, adaptive 100x21 is
    $4\times10^{-4}$ off (17 s; the same as uniform 200x41). Adaptive 200x61 is $2\times10^{-5}$ off,
    10x better than uniform 400x61.
  - **Not converged.** Near the second branch's $\omega_{\max}\approx0.198$ and on the third branch,
    the two grids differ by 1–4%. The core is then a $\phi$ spike at $r\sim0.03$ with $f(0)\sim10^{-4}$.
    The descending runs continue there until a stop condition fires (150x41: first law at
    $\omega=0.161$; 200x61: step size at $\omega=0.177$), so those last ~15–20 models are not
    reliable.
  - **Cost.** The 200x61 adaptive factorization needs more memory than uniform 400x61, because the
    clustering adds fill-in.
- **Deep family with `ilut`** (`runs/deep/`: `segments.py`, `plot_deep.py`): adaptive 300x61 (1347 s,
  0.78 GB) and 400x81 (3878 s, 1.13 GB), descending legs, after the log-radius wall detection and
  the first-law retries. Before those two changes both grids stopped at $\omega\approx0.175$, on a
  map jump.

  | quantity | 300x61 | 400x81 |
  |---|---|---|
  | $\omega_{\min}$ | 0.12983 | 0.12984 |
  | $M_{\max}$ ($Q$) | 7.6307 (35.81) | 7.6338 (35.83) |
  | $C^K_{\max}$ | 0.4162 | 0.4162 |
  | $\omega_{\max}$ of the second branch | 0.19825 | 0.19818 |
  | end of the third branch: $\omega$, $M$, $C^K$, $\max\phi$ | 0.1676, 5.049, 0.393, 0.068 | 0.1698, 4.945, 0.393, 0.072 |

  Agreement between the two grids (median / max relative difference):
  - first branch: $M$ $8\times10^{-6}$ / $7\times10^{-4}$;
  - second branch: $2\times10^{-5}$ / $5\times10^{-4}$;
  - third branch down to $\omega\approx0.170$: $M$, $J/M^2$ and $C^K$ $3\times10^{-4}$ / $4$–$8\times10^{-3}$. The
    central spike amplitude $\max\phi$ converges only to 1–6% there.

  Light rings (`runs/lightrings/`, adaptive 150x41 and 200x61, `light_rings = 1`; `transitions.py`
  lists the onsets and `lr_check.py` re-derives the rings from saved profiles with a different
  derivative):
  - **Counter-rotating pair.** It appears on the first branch at $\omega\approx0.187$, $C^K\approx0.29$
    (both grids agree to the step spacing), before $\omega_{\min}$ and $M_{\max}$. The outer ring is
    born near $R\approx3.9M$ and settles at $3.82M$; Kerr with the same $J/M^2\approx0.7$ has its
    counter-rotating photon orbit at $R=3.82M$. Its stable partner moves deep into the core
    ($R\approx0.2M$).
  - **Co-rotating pair.** It appears on the third branch at $\omega\approx0.185$, $C^K\approx0.38$, inside
    the star ($R\approx0.3M$). Its two rings are close together, so their positions are only good to
    5–7%.
  - **Consistency.** Counts never flicker along the families, and the counter-rotating radii agree with
    the independent check to $\le0.2\%$.

  Both runs end where the discretization's own first law fails by ~2%, and halving the step does
  not reduce it ($f(0)\approx10^{-4}$, a $\phi$ spike at $r\approx0.04$). Going further needs finer core
  resolution.

### Proca and Dirac stars (`matter = proca | dirac`)

Free massive vector and spinor fields on the same Lewis–Papapetrou metric, following Herdeiro,
Perapechka, Radu & Shnir 2019 (arXiv:1906.05386; source in `papers/1906.05386_src/`). Units
$\mu = 1$, $G = 1$ ($\kappa = 8\pi$). Both models start directly at $\kappa_{\rm target}$, since
free fields have no flat-space solitons. A good seed is needed, see `seed_file` below.

**Metric equations** (`tools/derive_matter.py`). The scalar equations (Kleihaus B.2 plus the S&E
fixes) are fixed combinations $\sum c^X_{ab}(G_{ab}-\kappa T_{ab})$:
- $c^f$: $gl r^2/f,\ 2gWlr/f,\ g(W^2 l+f^2/\sin^2)/f$ on $tt, t\varphi, \varphi\varphi$, and $fr^2,\ f$ on $rr, \theta\theta$;
- $c^l$: $2lr^2,\ 2l$;
- $c^g$: $2g^2/\sin^2,\ -2g$;
- $c^W$: $2gr/\sin^2,\ 2gW/\sin^2$.

They are solved for symbolically, depend on the metric only, and reproduce the scalar equations
exactly (regression in the script). Swapping $T_{ab}$ gives the metric equations for any matter,
with the same elliptic structure.

**Proca** ($A = e^{i(\omega t+m\varphi)}(iV dt + H_1/r\,dr + H_2 d\theta + iH_3\sin\theta\,d\varphi)$,
fields `V H1 H2 H3`):
- **Equations.** The Proca equation itself, $\nabla_aF^a{}_b=\mu^2A_b$; its divergence gives
  $\nabla\cdot A = 0$ identically. Adding $\nabla_b(\nabla\cdot A)$ would make the system an elliptic
  vector wave equation, but it then admits $A=\nabla\chi$ with $\chi$ a Klein–Gordon bound state of the
  self-gravitating background. The solver found exactly such a spurious "scalar star" before this was
  removed.
- **Structure.** The $H_1$ and $H_2$ rows are curl-curl like ($H_{1,\theta\theta}-rH_{2,r\theta}$,
  $r^2H_{2,rr}-rH_{1,r\theta}$), so mixed-derivative stencils are needed (5×5 Jacobian box).
- **Parities.** $V, H_1 \sim (-1)^m$ and $H_2, H_3\sim(-1)^{m+1}$ on the axis; $H_2$ odd on the equator.
- **Diagnostic.** The Lorenz condition is reported as the `lorenz` column, max $|\nabla\cdot A|$ over
  max $\sum|$terms$|$.
- **Validation.**
  - Static spherical stars ($m=0$): $M_{\max}=1.0575$ at $\omega=0.878$; Brito, Cardoso, Herdeiro &
    Radu 2016 give 1.058 at 0.875.
  - $m=1$: compared with the paper's Table 1 below.
- **Linear solver.** With $m=0$ the $H_1$ row is algebraic and its diagonal passes through zero, so
  ILUT fails and `linear_preconditioner = lu` is needed. For $m=1$, ILUT works.

**Dirac** ($\Psi=e^{i(m\varphi-\omega t)}(\psi_1,\psi_2,-i\psi_1^*,-i\psi_2^*)$, $\psi_1 = P+iQ$,
$\psi_2 = X+iY$, $m = $ `m_azimuthal`/2; `tools/derive_dirac.py`):
- **Representation.** The ansatz is consistent only in a suitable $\gamma$ representation. Writing it
  as the fixed set of $u\mapsto -iS\bar u$ ($S$ swaps the 2-spinors) requires
  $S\bar\gamma^aS^{-1} = -\gamma^a$ for the legs carrying the phase ($t,\varphi$) and $+\gamma^a$ for
  $r,\theta$. The chiral form with $\sigma_1,\sigma_3,\sigma_2$ on $e_1,e_2,e_3$ satisfies this. The
  script checks $E_{\rm lower}=-i\,\overline{E_{\rm upper}}$ to $10^{-16}$ at curved-metric points;
  the standard chiral and Dirac forms fail.
- **Covariance check.** The spin connection $\Gamma_\mu = \frac14\omega_{\mu ab}\gamma^a\gamma^b$ is
  verified to be flat in flat space.
- **Unknowns.** `D1 D2 D3 D4` $=(P\mp X)/\sqrt2,\ (Q\mp Y)/\sqrt2$, in which the pole conditions are
  parities: $D_2 = D_4 = 0$ on $\theta = 0$, $D_1, D_3$ even. The rows are Cauchy–Riemann pairs in
  $(\ln r,\theta)$.
- **Stress tensor.** $T_{ab}=-\frac i2[\bar\Psi\gamma_{(a}D_{b)}\Psi - D_{(a}\bar\Psi\gamma_{b)}\Psi]$ with
  $\bar\Psi = -\Psi^\dagger\gamma^0$, which gives positive $j^t$ and energy.
- **Three discretization points**, each found by a failed consistency check:
  1. **Checkerboard modes.** Centred differences of a first-order system leave the node-to-node
     sawtooth free. Each Cauchy–Riemann pair therefore uses forward $\partial_r$ with backward
     $\partial_\theta$ in one row and the reverse in the other, which behaves like a staggered grid.
     Fourth-difference dissipation (`dirac_dissipation`) was tried and is off by default.
  2. **Translation zero mode and reflection.** On the full $\theta\in[0,\pi]$ the star can slide
     along the axis, and Newton stalled on that zero mode. The solver therefore uses $[0,\pi/2]$ with
     the equatorial reflection $D_1(\pi-\theta)=sD_4(\theta)$, $D_2(\pi-\theta)=sD_3(\theta)$
     (`eq_partner_`; `dirac_parity` $= s = \pm1$). Because the mass terms carry the signs $+,-,-,+$
     in rows $D_1..D_4$, $D_1\leftrightarrow D_2$ is *not* a symmetry. A sympy check maps each full
     curved row onto its partner's row for the right pairing and none for the wrong one. A first
     version used $D_1\leftrightarrow D_2$; its runs, a spurious two-lobed configuration, are in
     `runs/dirac/obsolete_wrong_reflection/`.
     - $s=-1$: the ground state, an s-wave-like star with its density peaked at the centre and
       nearly spherical.
     - $s=+1$: a p-like state with opposite parity and a spherical-shell density.
  3. **Origin.** In flat space the half-angle pattern $D_1:D_2=D_3:D_4=\cos\frac\theta2:\sin\frac\theta2$
     has radial exponents $0$ and $-2$, so the regular spinor is *nonzero* at $r=0$. The
     $r$-weighted equations at $r=0$ lose $r\partial_r$ and fix only that ratio, not the amplitude;
     on the grid they pin $D=0$. The origin rows are therefore smoothness along rays: quadratic
     extrapolation from $i=1,2,3$. `dirac_origin_zero = 1` imposes the paper's $P=Q=X=Y=0$ instead.
- **Sign of $J$.** $J$ comes out negative ($J = -Q/2$), because the Dirac phase is the conjugate of
  the scalar/Proca convention.

**Results** (`runs/proca/m1`, `runs/dirac`, figure `runs/proca_dirac.png` from `runs/plot_proca_dirac.py`):

| | $M_{\max}$ (at $\omega$) | $J_{\max}$ | $\omega_{\min}$ | $M, J$ at $\omega_{\min}$ | $M=\mu Q$ at $\omega$ |
|---|---|---|---|---|---|
| Proca $m=1$, paper | 1.125 (0.562) | 1.259 | 0.469 | 1.086, 1.180 | – |
| Proca $m=1$, adaptive 200×41; end restarted on 250×41 | 1.1238 (0.546) | 1.2567 | ≤ 0.4661 (no turn yet) | at 0.469: 1.089, 1.186 | – |
| Dirac $m=\frac12$, paper ÷ 2 | 0.7545 (0.795) | 0.3945 | 0.680 | 0.599, 0.2845 | 0.6515 (0.692) |
| Dirac ground state ($s=-1$), 160×21 / 240×31 | 0.7539 / 0.7543 (0.80) | 0.3942 / 0.3944 | 0.6796 / 0.6795 | 0.586, 0.275 | 0.664 (0.696) |
| Dirac p-like ($s=+1$), 160×21 / 240×31 | 1.3075 (0.810) | 0.683 | 0.7000 | 1.008, 0.476 | 1.143 (0.712) |

- **Consistency.** On both families $J = mQ$ to $10^{-5}$ (Proca) and $|J| = Q/2$ to $10^{-4}$ (Dirac,
  first branches). For Dirac the Komar mass equals the ADM mass to $10^{-5}$, and the Proca Lorenz
  residual stays ≲ $10^{-4}$. A family stops when the first law $dM=\omega\,dQ$ is violated by more than
  `first_law_tol`.
- **Proca.** The family passes $\omega = 0.469$, where the paper stopped.
  - **200×41:** the first-law guard (3%) ends it at $\omega=0.4669$.
  - **250×41 restart** (`init_file`, `runs/proca/m1/hires_ext_250x41`, `hires_hop1`): restarted from
    the 200×41 model at $\omega=0.46898$, it continues to $\omega=0.4661$ ($M=1.0820$) and is still
    descending. The higher resolution lowers the first-law residual near the end from 1.4–1.9% to
    0.2–0.8%. $M$ agrees with 200×41 to $1$–$3\times10^{-4}$, and the Komar–asymptotic mass mismatch
    at fixed $\omega$ shrinks; it is $4\times10^{-3}$ at the ends of both grids.
  - **What stops it.** Arclength steps collapse ("step size underflow") once the core is ultra-compact:
    the first radial node is at $r\approx3\times10^{-4}$, and successive adaptive regrids perturb the
    secant. Restarting from the last model gains a few more models per hop.
  - **Memory.** 300×41 and 300×61 need more than 3 GB, mostly in the float-LU fallback, which is
    beyond this machine. 250×41 peaks at 2.1–3 GB.
- **Ergoregions** (`ergoregion = 1`; stars in the figure; onset interpolated where $\max g_{tt}$ turns
  positive):
  - **Proca $m=1$:** at $\omega=0.603$, $M=1.115$, before $M_{\max}$ and after the light rings
    (which appear at $\omega\approx0.71$). Near the end of the family $\max g_{tt}$ reaches 0.8, and
    the equatorial ergosurface extends from $0.02M$ to $2.4M$.
  - **Dirac ground state:** just past $\omega_{\min}$, at $\omega=0.683$, $M=0.560$, on the second
    branch. It first appears at $0.76M$–$1.56M$ on the equator.
  - **Dirac p-like state:** none on the traced part; $\max g_{tt}$ comes within $-0.008$ near
    $\omega=0.877$ and then recedes.
- **All three families** (`runs/onsets.png` from `runs/plot_onsets.py`): $M(\omega)$ for the solitonic
  scalar ($\sigma=0.05$, $m=1$; `runs/lightrings/a200x61_ergo`), Proca and Dirac (ground state)
  families, with the onsets of light rings (circles) and ergoregions (stars):

  | family | light rings appear | ergoregion appears |
  |---|---|---|
  | scalar, solitonic | $\omega=0.188$, $M=4.62$ (before $M_{\max}$) | $\omega=0.134$, $M=6.93$ (second branch, past $\omega_{\min}=0.1298$) |
  | Proca $m=1$ | $\omega=0.713$, $M=1.050$ | $\omega=0.603$, $M=1.115$ (before $M_{\max}$) |
  | Dirac ground state | $\omega=0.703$, $M=0.680$ | $\omega=0.683$, $M=0.560$ (second branch, past $\omega_{\min}=0.680$) |

  In every family the light rings come first. Only for Proca does the ergoregion appear on the
  first branch.
- **Dirac ground state.** The paper's frequencies are reproduced exactly. Its $M$, $J$, $Q$ are all
  **twice** ours, i.e. a different effective $G$. Ours has the right weak-field limit: as
  $\omega\to\mu$ the ground state tends to the Schrödinger–Newton mass
  $M_{\rm SN}=\sqrt{(1-\omega)/0.16276}$ (ratio 0.998 at $1-\omega = 1.3\times10^{-3}$), as does the
  Proca $m=1$ family (ratio 1.0000 at $1-\omega=2\times10^{-5}$). A single spin-½ particle species in
  its s-state must have that limit. The paper's earlier spherical Dirac stars use two spinors, which
  would explain the factor 2. $M$ at $\omega_{\min}$ is interpolated through the turning point; the
  paper's value is 2% higher, consistent with reading off a nearby model.
- **Shapes.** The paper describes its Dirac stars as toroidal and imposes $P=Q=X=Y=0$ at $r=0$. Our
  ground state has a nonzero, almost spherical, centrally peaked density. With `dirac_origin_zero = 1`
  the solver converges to the same two stars at $\omega=0.9$ ($M$ equal to $2\times10^{-5}$), with a thin
  layer at the origin and no separate toroidal family. That is expected: the regular spinor is
  nonzero at $r=0$, so on the grid their condition acts only as a one-node defect. The two parities give two
  separate families with their own spirals; the paper reports only one.

**Seeds** (`seed_file`). A radial profile `r f l g W` plus either all matter columns, or one
amplitude column $\psi$ that is spread over the $m\ge1$ Proca pattern ($\psi(e_x+ie_y)$:
$H_1=r\psi\sin$, $H_2 = r\psi\cos$, $H_3=r\psi$, $V=-\psi'\sin/\omega$) or the Dirac s-wave pattern.
The working seeds are the 1D mini boson star at $\omega=0.9$ (1D solver, `bs.dat`), with
$\psi=\sqrt2\phi$ (Proca) or $\phi/\sqrt2$ (Dirac); see `runs/proca/`, `runs/dirac/`.
`tools/proca_spherical.py` is an independent shooting code for spherical Proca stars, and
`--seed` converts it to isotropic coordinates. It truncates the long $A_r$ tail, so its masses are a
few percent low; it is used only for seeds.

### Performance

Measured on this 16-core machine; every benchmark reproduces the original results (Q=410 Q-balls,
cusps, $M_{\max}$ to all printed digits; whole families within spline-interpolation error):

| test | original | now |
|---|---|---|
| m=1 Q-ball march to Q=410 (100x21) | 27 s | 4.0 s |
| single solve from seed (200x41) | 25 s | 3.9 s |
| kappa=0.3 n=1 family, 63 models (100x21) | 129 s | 15 s |
| mini m=1 family, 45 models (100x21) | 86 s | 11.7 s |
| mini m=1 family, 49 models (200x41) | 411 s | 57 s |
| kappa=0.1 n=1 family, 84 models (100x21) | 137 s | 15 s |

What does it (rotating solver):
- **Reuse of the factorization** (`jacobian_reuse`). Default `newton_linear = gmres_lu`: every
  Newton iteration assembles the fresh Jacobian and solves with GMRES preconditioned by the last
  sparse LU, refactoring only when GMRES needs > `gmres_lu_max_iters` (a family of ~50 models needs
  ~5 factorizations). `chord` uses the stale LU directly and refreshes it on slow convergence
  (similar speed at 100x21, slower at 200x41). Converged states are the same: only the path to
  R(x)=0 changes.
- **Stencil-local Jacobian**: each unknown is perturbed alone and only the ~10 nodes of its stencil
  are re-evaluated (parallel over grid rows; checked against a dense Jacobian with
  `BS_CHECK_JACOBIAN=1`); the |term| scales are skipped in these evaluations.
- **Quadratic predictor** through the last three solutions (`arclength_predictor`), ~20% fewer
  Newton iterations; the arclength constraint, hence each solution, is unchanged.
- `-O3`, OpenMP (residual on grids with >= 4000 nodes, Jacobian), `lu_pivot_threshold = 0.1`.
- Optional: `OMP_WAIT_POLICY=passive` (environment) gains another 4-10% by stopping idle OpenMP
  threads from spinning during the serial LU solves.

**Memory.** Diagnostics are switched on by environment variables:
- `BS_LU_STATS=1` prints the factorization kind, n, nnz(L+U) (exact LU kinds) and the process memory
  after each factorization.
- `BS_DUMP_JACOBIAN=file` writes the factored matrix; a `%03d` in the name keeps every one.
- `runs/memory/lubench.cpp` benchmarks factorizations and preconditioners on a dump.

The Jacobian itself is small: about 26 nonzeros per row, 10–25 MB. Almost all the memory is fill-in
of the sparse LU factors:
- 25x nnz(A) at 150x41 and 33x at 200x61, so it grows faster than the problem;
- up to 47x on the hard matrices near turning points, where threshold pivoting adds fill;
- the 150x41 factor is 190 MB normally and 560 MB at those spikes; the 200x61 factor is 500 MB at
  $\omega=0.303$ and makes up most of the 2 GB peak further along the family.

Measured alternatives (150x41 adaptive solitonic family; GMRES iterations to $10^{-10}$ on the
double-precision Jacobian):

| preconditioner | factor memory (normal / hardest) | GMRES its, fresh | its, reused on the next matrix |
|---|---|---|---|
| LU double, COLAMD (current) | 192 / 560 MB | 1 | 29–94 |
| LU float, COLAMD | 107 / 320 MB | 3–4 | 29–95 |
| ILUT fill 20, drop 1e-5 | 51 / 53 MB | 6–17 | 31–182 |
| ILUT fill 40, drop 1e-6 | 86 / 88 MB | 4–10 | 30–94 |
| LU double, AMD / grid nested dissection | 691 / 1245 MB | 1 | – |

- At 200x61 ($\omega=0.303$) the LU factor takes 501 MB, float LU 286 MB, and ILUT(20, 1e-5) 85 MB
  with 8 iterations.
- A reused factor, which is how `gmres_lu` normally works, needs about as many GMRES iterations with
  ILUT(40) as with the exact LU, so the exact factor's extra accuracy is mostly wasted.
- This led to the `linear_preconditioner` option below.

**`linear_preconditioner`** chooses the factorization behind `gmres_lu`:
- `ilut` (default): incomplete LU with `ilut_fill` (20) and `ilut_droptol` (1e-5).
- `lu`: exact double-precision LU, the solver's behaviour before this option. After a fresh
  factorization the Newton step is a direct solve.
- `lu_float`: the same LU in single precision.

For the two inexact kinds, the step after a fresh factorization is solved by GMRES, allowed up to
3x `gmres_lu_max_iters` iterations. A factorization that breaks down, or whose GMRES does not
converge, climbs a fallback ladder:
- `ilut`: ILUT, then ILUT with 2x fill and droptol/10, then float LU;
- `lu_float`: float LU, then exact LU.

The run summary counts the fallbacks. Chord and plain Newton (`newton_linear = chord`,
`jacobian_reuse = 0`) apply the factor as the Newton step, so they always use `lu` and print a
warning if something else was set.

Benchmark (`runs/precond/`: `setup.sh`, `run.sh`, `analyze.py`): solver-clock time and peak
memory, each run capped at 2.5 GB without swap. The baseline is the solver as it was before this
option.

| case | baseline (`lu`) | `lu_float` | `ilut` (default) |
|---|---|---|---|
| repulsive, one solve from the seed, 200x41 | 3.4 s, 434 MB | 3.3 s, 259 MB | 1.8 s, 128 MB |
| KKLS $\omega$ march, 200x41 | 63 s, 400 MB | 91 s, 281 MB | 35 s, 233 MB |
| solitonic family (adaptive) 150x41 | 340 s, 705 MB | 605 s, 607 MB | 362 s, 227 MB |
| axionic $\omega$ march, 400x61 | 322 s, 1553 MB | 431 s, 1060 MB | 157 s, 744 MB |
| solitonic family (adaptive) 200x61 | 802 s, 2026 MB | 1048 s, 1185 MB | 786 s, 1220 MB (1 fallback) |
| solitonic $\omega$ march (adaptive) 400x81 | killed at 2.5 GB | 593 s, 1802 MB | 226 s, 997 MB |

- **Robustness.** Every run that fit in memory completed, and every result agrees with the
  baseline. Runs that took the same steps agree to $\le10^{-10}$. Runs whose step sizes differed sit
  on slightly different adaptive grids: their median deviation is $\le5\times10^{-6}$, with
  $\sim10^{-3}$ only at the fold and on the unconverged third branch.
- **The fallback works.** It fired once, on the hardest 200x61 matrix. That single float LU set the
  run's peak memory; the benchmark predates the intermediate stronger-ILUT rung, which now comes
  first. With `ilut_fill = 40, ilut_droptol = 1e-6` there was no fallback (516 MB) but
  the run took 1072 s. A forced test (`ilut_fill = 1`) with fallbacks on half the factorizations
  also completed with the right answer.
- **`ilut` speed.** It is fastest on large and smooth problems (axionic 2.1x, 400x81 >2.6x against
  `lu_float`). On the hard adaptive families it roughly ties with the baseline.
- **`lu_float` is not worth using on its own.** It halves the factor, but the inexact Newton steps
  cost 15–20% more Newton iterations and its stale factor degrades faster, so it is slower everywhere.
- **Remaining memory.** With `ilut` about half the peak is the factor. The rest is several
  simultaneous copies of the Jacobian (assembly buffers, the CSR for GMRES, Eigen's matrix inside
  the factorization, triplets) plus up to 75 Krylov vectors. Limiting glibc's malloc arenas
  (`MALLOC_ARENA_MAX=2`) takes another ~20% off small runs at a ~7% time cost.

Tried and not adopted: AMD ordering (5x more fill), geometric nested dissection (more fill with
Eigen's SparseLU than COLAMD), an adaptive refactor schedule (no gain). Where the time goes now
(200x41 family): ~80% in the LU triangular solves inside GMRES (~19 ms each), ~12% LU
factorizations. A faster sparse direct solver (SuiteSparse UMFPACK, MKL PARDISO) is the next lever.
For coarser grids at the same accuracy, see `radial_grid = adaptive`.

## Next implementation milestones

1. Rotating:
   - S&E m=2 models and the other potentials' full families on the adaptive grid;
   - the deep spiral beyond the second branch, which needs a finer core, or a $\theta$ grid adapted
     near the axis;
   - speed: UMFPACK/PARDISO (LU is ~80% of the time).
2. `continuation_mode = lapse` is ill-posed as formulated (Dirichlet $f(0)$ replaces regularity
   $\partial_r f(0)=0$, and $\rho$ gets Neumann conditions at both ends) and does not converge.
3. The 1D solver still uses the uniform $\bar r$ grid. The `RadialMap` of the rotating solver could be
   reused there.
5. `omega`/`charge` modes are kept for single solves at fixed $\omega$ or $Q$; their family
   continuation (switching at turning points) retraces branches and is superseded by `arclength`.
4. Add regression tests against published spherical sequences.

## Quick try

```bash
make
make run
```

You can also specify another parameter file:

```bash
./rotating_bs_solver path/to/params.par
```

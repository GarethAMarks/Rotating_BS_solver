#ifndef BS_PARAMS_H_
#define BS_PARAMS_H_

#include <string>

struct BSParams {
    // General run controls
    bool quiet = false;
    std::string sequence_mode = "spherical"; // spherical | rotating
    std::string continuation_mode = "arclength"; // arclength | omega | charge | lapse

    // Physical model
    std::string potential = "mini"; // mini | kkls | solitonic | axionic | repulsive
    std::string matter = "scalar";   // rotating solver: scalar | proca | dirac (free fields of mass mu)
    // Dirac: optional fourth-difference dissipation sigma in each first-order row (r-weighted rows:
    // -sigma (r xi'/h_xi) delta^4_xi D - sigma (1/h_theta) delta^4_theta D). Not needed with the biased
    // (staggering-like) derivative stencils the Dirac rows use; kept for experiments.
    double dirac_dissipation = 0.0;
    // Dirac origin rows: 0 = the field equations at r = 0 (regular spinor, nonzero there; s-wave-like,
    // spheroidal stars); 1 = P = Q = X = Y = 0 at r = 0 as imposed by Herdeiro et al. 2019 (toroidal stars).
    int dirac_origin_zero = 0;
    // Dirac equatorial reflection D1(pi - theta) = s D4(theta), D2(pi - theta) = s D3(theta): s = +1 or -1.
    int dirac_parity = -1;  // ground state
    double mu = 1.0;
    double kappa = 0.0;      // gravitational coupling continuation parameter
    double sigma = 0.05;     // solitonic coupling
    double kkls_kappa = 0.1; // KKLS coupling
    double f_axion = 0.005;  // axionic coupling
    double lambda_rep = 0.0; // repulsive quartic coupling
    // Kleihaus et al. sextic potential U = lambda (phi^6 - a phi^4 + b phi^2), boson mass sqrt(lambda b)
    double kleihaus_lambda = 1.0;
    double kleihaus_a = 2.0;
    double kleihaus_b = 1.1;

    // Target parameters (one typically imposed at infinity)
    double omega_target = 0.95;
    double charge_target = 0.0;

    // Grid / numerics
    int Nr = 401;
    double rbar_min = 0.0;
    double rbar_max = 1.0;
    int fd_order = 5;
    int max_newton_iters = 50;
    double abs_tol_residual = 1e-10;
    double abs_tol_update = 1e-12;
    double linear_tol = 1e-8;
    double line_search_min = 1e-8; // allow smaller backtracks for stiff steps
    int max_line_search_steps = 40; // allow more backtracks when needed
    double armijo_c = 1e-3; // Armijo sufficient-decrease constant (was hard-coded)
    int gmres_restart = 30;
    int gmres_max_iters = 200;
    double jacobian_fd_eps = 1e-7;
    // Converged when |R_i| <= rel_tol_residual * scale_i + abs_tol_residual for
    // every row (scale_i = sum of |terms| in row i). A Newton stall (line search
    // failure, vanishing update) is accepted if the same holds with stall_rel_tol.
    double rel_tol_residual = 1e-12;
    double stall_rel_tol = 1e-9;
    // Converged states with max|phi| below this are rejected as the trivial vacuum.
    double trivial_phi_tol = 1e-8;

    // Predictor smoothing iterations applied before Newton on continuation steps
    int predictor_smooth_iters = 3;

    // Continuation controls
    double delta_kappa = 0.01;
    double delta_omega = 0.002;
    // Continuation by central lapse: delta_lapse is the initial step in
    // sqrt(f(0)) (the central lapse). lapse_target is the target value of
    // sqrt(metric_f[0]) used in the 'lapse' continuation mode.
    double delta_lapse = 1e-3;
    double lapse_target = 0.0;
    double delta_charge = 0.01;
    // Pseudo-arclength continuation (continuation_mode = arclength)
    int arclength_max_steps = 500;
    double arclength_ds_max = 0.0;    // 0: 10x the first step length
    double arclength_min_cos = 0.8;   // min cosine between consecutive secants
    double arclength_omega_min = 0.0; // stop once omega drops to this value
    double arclength_omega_max = 0.0; // stop once omega rises to this value (0: the boson mass)
    double core_resolution_tol = 0.1;  // stop once f(r_1)/f(0) - 1 exceeds this (core unresolved)
    double wall_resolution_tol = 0.2;  // stop once max_i |phi_{i+1} - phi_i| / max|phi| exceeds this
    int first_law_retries = 2;         // halve the step this many times before stopping on a violation
    double first_law_tol = 0.05;       // rotating: stop once consecutive models violate dM = omega dQ by more
                                       // than this (relative; 0 disables)
    // Target kappa to continue to (default: 8*pi ~ natural units G=1)
    double kappa_target = 25.132741228718345; // 8*pi

    // If true, append per-step branch data to `initial_models.dat` during continuation
    bool write_bsdata_per_step = true;

    // Q-ball shooting bootstrap controls (kappa = 0)
    bool use_qball_bootstrap = true;
    double qball_rmax = 80.0;
    double qball_dr = 0.01;
    double qball_phi0_min = 1e-4;
    double qball_phi0_max = 2.0;
    int qball_scan_points = 80;
    double r_match_fac = 0.9;

    // Rotating (2D) solver, sequence_mode = rotating
    int Ntheta = 24;               // theta grid points on [0, pi/2]
    int m_azimuthal = 1;           // azimuthal number m (Kleihaus' n)
    double rot_seed_amp = 1.0;     // seed phi = amp * (rho/R)^m exp(m/2 (1 - rho^2/R^2) - z^2/(2 R^2))
    double rot_seed_width = 3.0;   // seed radius R (rho = r sin(theta), z = r cos(theta))
    // Optional radial seed profile (rotating solver): columns r f l g W <matter fields>, radius r
    // increasing (isotropic); interpolated onto every theta ray (beyond the last r: f = l = g = 1,
    // W = 0, matter 0). With a single matter column psi(r) and Proca m >= 1, the angular pattern of
    // the analytic seed is used with this psi. Overrides rot_seed_amp / rot_seed_width.
    std::string seed_file = "";
    // Restart (rotating solver): a bs2d.dat profile written by an earlier run (any grid, same matter
    // and m). Its fields are interpolated onto the new grid (adaptive grids are fitted to it first)
    // and its omega replaces omega_target. Overrides the seeds.
    std::string init_file = "";
    // With init_file: the model before it (bs2d_prev.dat of the same run). The restart then
    // re-converges both with the arclength constraint along their secant (well posed at folds)
    // instead of solving at fixed omega.
    std::string init_file_prev = "";
    // Single model (rotating solver): > 0 builds only the model with phi_max = this value (secant on
    // omega from omega_target, at kappa_target), writes bs2d.dat and stops; no continuation.
    double single_model_phi_max = 0.0;
    double rot_omega_stop = 0.3;   // omega march: stops exactly at this omega
    int rot_max_steps = 200;       // omega march: maximum number of steps
    // Radial grid (rotating solver): uniform in xi = map(r). "compactified": xi = r/(L+r) with
    // L = radial_scale. "adaptive": a fraction radial_cluster_fraction of the points clustered on
    // the outer wall of phi, L = radial_scale_ratio * (outer edge of the wall region); the map
    // follows the solution and the states are regridded when it drifts by more than regrid_tol.
    std::string radial_grid = "compactified";
    // Rotating solver: find equatorial light rings for every model and add their number as a
    // "light_rings" column to families2d.dat (radii in the log).
    bool light_rings = false;
    // Rotating solver: ergoregion check (g_tt > 0) for every model: adds the column ergo_gtt_max
    // (max of g_tt over the grid; > 0 iff an ergoregion exists) to families2d.dat, equatorial
    // ergosurface radii in the log.
    bool ergoregion = false;
    double radial_scale = 1.0;
    double radial_cluster_fraction = 0.4;
    double radial_scale_ratio = 0.5;
    double regrid_tol = 0.1;
    bool jacobian_reuse = true;    // chord Newton: reuse the factored Jacobian (see newton_solve)
    double chord_rate = 0.25;      // refresh the Jacobian when |R| drops by less than this factor
    // Linear solves with jacobian_reuse: "chord" (reuse the stale LU as the Jacobian) or
    // "gmres_lu" (fresh Jacobian each iteration, GMRES preconditioned by the stale LU).
    std::string newton_linear = "gmres_lu";
    int gmres_lu_max_iters = 25;   // refactor when GMRES needs more iterations than this
    // Factorization behind gmres_lu: "ilut" (incomplete LU, 2-3x less memory than lu; a
    // factorization that breaks down or does not precondition well enough is redone as a
    // stronger ILUT, then as a float LU), "lu" (exact, double) or "lu_float" (exact LU in single
    // precision; slower than both). Chord / plain Newton always use "lu".
    std::string linear_preconditioner = "ilut";
    double ilut_fill = 20.0;       // ilut: max entries per row of L and U, in units of the row's nonzeros in J
    double ilut_droptol = 1e-5;    // ilut: drop entries below this x the row norm
    // ilut: keep the factors in single precision (1: about half the memory and faster, but near the
    // end of strong-gravity families float round-off spoils the preconditioner and the continuation
    // stalls) or double (0)
    int ilut_float = 0;
    // ilut: last rung of the fallback ladder, a single-precision exact LU (1). It needs several
    // times the ILUT's memory (2-3 GB at 250x41); 0 skips it, so a matrix that no ILUT rung can
    // precondition fails the Newton step and the continuation halves its step instead.
    int ilut_lu_fallback = 1;
    double gmres_lu_tol = 1e-6;    // GMRES relative residual tolerance (inexact Newton; result unaffected)
    double lu_pivot_threshold = 0.1; // sparse LU partial pivoting threshold (< 1 prefers the diagonal: less fill)
    std::string arclength_predictor = "quadratic"; // Newton seed in 2D arclength: quadratic | linear

    // Output
    std::string output_prefix = "out/bs";
    // Linear solver choice: "banded" (colored FD Jacobian + banded LU),
    // "direct" (dense FD Jacobian + dense LU; slow, kept for cross-checks)
    // or "gmres" (matrix-free Newton-Krylov, unpreconditioned)
    std::string linear_solver = "banded";
};

#endif // BS_PARAMS_H_

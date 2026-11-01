#ifndef ROTATING_SOLVER_H_
#define ROTATING_SOLVER_H_

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bs_params.h"
#include "linear_solvers.h"

// Stationary axisymmetric (rotating, azimuthal number m) boson stars / Q-balls on a
// (xi, theta) grid: xi in [0,1] uniform with a compactifying radial map xi(r) (RadialMap;
// default xi = r/(1+r)), theta in [0, pi/2] (equatorial symmetry).
// Field equations: Kleihaus et al. (2005) App. B.2 with the Siemonsen & East (2021) App. A
// corrections, generated and verified by tools/derive_rotating.py.
//
// Unknowns: f, l, g, W, phi at every node (interleaved per node, theta fastest) and the
// frequency omega as one scalar, closed by one constraint row (fixed omega or fixed Q).
// Radial map r -> xi in [0, 1): a compactification r/(L+r) plus point clusters k, each a top hat
// on [ra, rb] with tanh edges of width delta, carrying the fraction beta_k / (1 + sum beta):
//   xi(r) = (r/(L+r) + sum_k beta_k B_k(r)/B_k(inf)) / (1 + sum beta),
//   B_k' = (tanh((r-ra)/delta) - tanh((r-rb)/delta))/2.
// Smooth and analytic (xi, xi', xi''), so finite differences in xi stay 4th order. L = 1 without
// clusters is the original map xi = r/(1+r). Cluster 0: core (symmetric about r = 0); cluster 1:
// the star's outer wall.
struct RadialMap {
    struct Cluster {
        double beta = 0.0, ra = 0.0, rb = 1.0, delta = 1.0;
        double B(double r) const;
        double B_inf() const;
    };
    double L = 1.0;
    std::array<Cluster, 2> cl;
    double xi(double r) const;
    void derivs(double r, double& xi_r, double& xi_rr) const;  // d xi/dr, d^2 xi/dr^2
    double r_of_xi(double xi) const;                           // inverse (xi < 1)
    double beta_sum() const { return cl[0].beta + cl[1].beta; }
    double fraction(int k) const { return cl[static_cast<size_t>(k)].beta / (1.0 + beta_sum()); }
};

// Light rings (circular photon orbits) in the equatorial plane, from the equatorial profiles of
// f, l, W at increasing r > 0. On the equator of the Lewis-Papapetrou metric
// g_tt = -f + l W^2/f, g_tphi = -l r W/f, g_phiphi = l r^2/f, so g_tphi^2 - g_tt g_phiphi = l r^2 and
// the angular velocities of tangential photons are H_+- = W/r +- f/(r sqrt(l)) (Cunha, Berti &
// Herdeiro 2017). Light rings are the critical points of H_+-: located on the grid as sign changes
// of the differences of H, refined by a parabola in the node index. Reported as circumferential
// radii r sqrt(l/f), split into co- and counter-rotating with the star (frame dragging W/r sets
// the sense of rotation; W = 0: all are counted as co-rotating).
struct LightRings {
    std::vector<double> R_co, R_counter;  // circumferential radii, inner first
    int count() const { return static_cast<int>(R_co.size() + R_counter.size()); }
};
LightRings find_light_rings(const std::vector<double>& r, const std::vector<double>& f, const std::vector<double>& l,
                            const std::vector<double>& W);

// Ergoregion: where the stationary Killing vector d_t is spacelike, g_tt = -f + l sin^2(theta) W^2 / f > 0.
// gtt_max is the largest g_tt on the grid (> 0 iff there is an ergoregion; for these stars it is a
// torus crossing the equator); R holds the circumferential radii r sqrt(l/f) of the equatorial
// ergosurface crossings (sign changes of g_tt, linear interpolation), inner first.
struct Ergoregion {
    double gtt_max = -1.0;
    std::vector<double> R;
    bool present() const { return gtt_max > 0.0; }
};
Ergoregion find_ergoregion(const std::vector<double>& r, const std::vector<double>& f, const std::vector<double>& l,
                           const std::vector<double>& W, double gtt_max);

class RotatingSolver {
public:
    explicit RotatingSolver(const BSParams& params);
    bool run();

private:
    using Vec = std::vector<double>;
    // Unknowns per node: the metric f, l, g, W, then the matter functions -- scalar: phi;
    // Proca: V, H1, H2, H3 (A = e^{i(w t + m varphi)} (i V dt + H1/r dr + H2 dtheta + i H3 sin dvarphi)).
    static constexpr int NF_MAX = 8;
    enum Field { F_f = 0, F_l = 1, F_g = 2, F_W = 3, F_p = 4, F_V = 4, F_H1 = 5, F_H2 = 6, F_H3 = 7,
                 F_D1 = 4, F_D2 = 5, F_D3 = 6, F_D4 = 7 };
    // Dirac: Psi = e^{i(m varphi - w t)} (psi1, psi2, -i psi1*, -i psi2*), psi1 = P + i Q, psi2 = X + i Y
    // (Herdeiro, Perapechka, Radu & Shnir 2019), solved for D1 = (P - X)/sqrt2, D2 = (P + X)/sqrt2,
    // D3 = (Q - Y)/sqrt2, D4 = (Q + Y)/sqrt2 on theta in [0, pi/2] with the equatorial reflection
    // D1 <-> +-D4, D2 <-> +-D3 (the full [0, pi] admits the z-translation zero mode of the star);
    // m = m_azimuthal / 2 (half-integer). See tools/derive_dirac.py.
    enum class Matter { Scalar, Proca, Dirac };
    bool theta_full_ = false;  // theta in [0, pi] (two poles) instead of [0, pi/2] (axis + equator)
    int je_ = 0;               // equator index
    double mval_ = 0.0;        // azimuthal number used in the equations (m_azimuthal / 2 for Dirac)
    Matter matter_ = Matter::Scalar;
    int nf_ = 5;
    // Parity of each field under theta -> -theta (axis ghosts) and theta -> pi - theta (equator
    // ghosts; with theta_full_ the second pole theta = pi): +1 even, -1 odd. Odd fields get Dirichlet
    // rows on that boundary, even ones d_theta = 0 (poles) or the field equation with mirrored
    // ghosts (equator). origin_neumann_:
    // d_r = 0 at r = 0 instead of the value 0 (m = 0 fields that are nonzero at the centre).
    int axis_sign_[NF_MAX] = {1, 1, 1, 1, 1, 1, 1, 1};
    int eq_sign_[NF_MAX] = {1, 1, 1, 1, 1, 1, 1, 1};
    bool origin_neumann_[NF_MAX] = {false, false, false, false, false, false, false, false};
    // Equatorial reflection partner: the ghost of field a beyond the equator is eq_sign_[a] times
    // field eq_partner_[a] at the mirrored node (Dirac: D1 <-> D4, D2 <-> D3; otherwise a itself).
    // Partnered fields: on the equator the second of each pair gets the row  field - partner = 0.
    int eq_partner_[NF_MAX] = {0, 1, 2, 3, 4, 5, 6, 7};
    void setup_matter();
    enum class Constraint { Omega, Charge, Arclength };

    struct Observables {
        double omega = 0, phi_max = 0, mass_komar = 0, mass_asym = 0, charge = 0, angmom = 0, angmom_asym = 0;
        double lorenz = 0;            // Proca: max |nabla.A| / max sum|terms| (should be ~ truncation error)
        double R99_K = 0, R99_Q = 0;  // circular radii containing 99% of M / Q (Siemonsen & East)
        LightRings lr;                // equatorial light rings (light_rings = 1 only)
        Ergoregion ergo;              // ergoregion (ergoregion = 1 only)
    };

    BSParams p_;
    int Nr_, Nt_, m_;
    double hr_, ht_;
    RadialMap map_;
    bool map_adapted_ = false;  // adaptive grid: map_ has been fitted to a solution
    int n_regrids_ = 0;
    Vec rbar_, r_, theta_;  // rbar_: the uniform xi nodes
    Vec xi_r_, xi_rr_;      // d xi/dr, d^2 xi/dr^2 at the radial nodes (0 at r = infinity)
    Vec x_;                 // current state
    double kappa_ = 0.0;
    Constraint constraint_ = Constraint::Omega;
    double omega_target_ = 0.0, charge_target_ = 0.0;
    Vec wq_;                // quadrature weights per node (dr dtheta dvarphi, both hemispheres)
    Vec wq_th_;             // theta-and-varphi weights (both hemispheres) per theta node
    int last_iterations_ = 0;
    int last_iters_to_floor_ = 0;
    double last_residual_ = 0.0;
    Vec arc_t_, arc_pred_;  // arclength: unit secant tangent and predictor (full state incl. omega)
    std::unique_ptr<SparseLUSolver> lu_;  // factored Jacobian, reused across iterations/steps
    bool lu_valid_ = false;
    Constraint lu_constraint_ = Constraint::Omega;
    long n_factorizations_ = 0, n_newton_its_ = 0, n_gmres_its_ = 0;
    std::string precond_ = "lu";      // effective linear_preconditioner
    long n_level_[3] = {0, 0, 0};       // factorizations per fallback level (see factor_entries)
    double t_jac_ = 0.0, t_lu_ = 0.0;  // accumulated timings (reported by run())
    int last_refreshes_ = 0;

    size_t idx(int i, int j, int a) const { return (static_cast<size_t>(i) * Nt_ + j) * nf_ + a; }
    size_t n_unknowns() const { return static_cast<size_t>(Nr_) * Nt_ * nf_ + 1; }
    // Matter amplitude at a node (|phi|; Proca: sqrt(V^2 + H1^2 + H2^2 + H3^2)): drives the
    // adaptive grid, the resolution indicators and phi_max.
    double amp(const Vec& x, int i, int j) const;
    // Physical derivatives at node (i, j) of field a: d/dr, d^2/dr^2 (and the |stencil| sums used
    // for rounding scales), d/dtheta, d^2/dtheta^2, and the mixed d^2/dr dtheta. At i = 0 the
    // radial derivative is one-sided (second order).
    void deriv_r(const Vec& x, int i, int j, int a, double& d1, double& d2, double& a1, double& a2) const;
    void deriv_t(const Vec& x, int i, int j, int a, double& d1, double& d2, double& a1, double& a2) const;
    void deriv_rt(const Vec& x, int i, int j, int a, double& d, double& ad) const;
    // Biased first derivatives (dir = +1 forward, -1 backward): 4th order on offsets -1..3 / -3..1,
    // lower order next to the radial boundaries. The Dirac rows pair them like a staggered grid
    // (see node_rows), which removes the checkerboard modes of centred differences.
    void deriv_r_biased(const Vec& x, int i, int j, int a, int dir, double& d1, double& a1) const;
    void deriv_t_biased(const Vec& x, int i, int j, int a, int dir, double& d1, double& a1) const;

    bool run_impl();
    void build_grid();  // nodes, map derivatives and quadrature weights for map_
    void seed_analytic();

    // Field value with parity ghosts in theta (axis: phi has parity (-1)^m, the rest even;
    // equator: all even).
    double val(const Vec& x, int i, int j, int a) const;

    // Residual and per-row scale (sum of |terms|, for the componentwise convergence test).
    void residual(const Vec& x, Vec& R, Vec* S) const;
    // Rows of one node (5 entries at Rn; scales at Sn if non-null).
    void node_rows(const Vec& x, int i, int j, double* Rn, double* Sn) const;

    // Observables by quadrature (+ asymptotic M, J from f and W when kappa > 0).
    Observables observables(const Vec& x) const;
    // Volume densities at a node (dr dtheta dvarphi, sqrt(-g) included): Komar mass, charge,
    // angular momentum; Proca also the Lorenz condition nabla.A and its sum of |terms|.
    void node_densities(const Vec& x, int i, int j, double& komar, double& charge, double& angmom,
                        double* lorenz = nullptr, double* lorenz_scale = nullptr) const;
    double charge(const Vec& x) const;
    void charge_gradient(const Vec& x, Vec& grad) const;

    // Newton with colored central-difference Jacobian and sparse LU.
    bool newton_solve(Vec& x);
    // Colored central-difference Jacobian at x (entries), its factorization into lu_, and both.
    bool assemble_jacobian(const Vec& x, std::vector<SparseLUSolver::Entry>& ent);
    // Factorization at a level of the fallback ladder of linear_preconditioner:
    //   ilut: ILUT (ilut_fill, ilut_droptol) -> ILUT (2 x fill, droptol / 10) -> float LU;
    //   lu_float: float LU -> exact LU;  lu: exact LU. False if it fails or the level does not exist.
    bool factor_entries(const std::vector<SparseLUSolver::Entry>& ent, int level = 0);
    int fallback_levels() const { return precond_ == "ilut" ? 3 : precond_ == "lu_float" ? 2 : 1; }
    bool factor_jacobian(const Vec& x);
    // GMRES on the Jacobian given by ent, preconditioned with lu_, at most max_iters iterations
    // (no restart); false if not converged.
    bool solve_gmres_lu(const std::vector<SparseLUSolver::Entry>& ent, const Vec& b, Vec& x, int& iterations,
                        int max_iters) const;
    bool within(const Vec& R, const Vec& S, double rel) const;

    // Pseudo-arclength continuation in the full state (as in the 1D solver).
    bool run_arclength(const std::string& fam);
    // Under-resolution indicators: f(r_1)/f(0) - 1 (core) and max cell jump of phi / max phi (wall).
    void resolution_indicators(const Vec& x, double& core, double& wall) const;

    // Adaptive radial grid (radial_grid = adaptive): the map wanted for the profile x (cluster on
    // the outer wall of phi over all theta rays, scale L from its outer edge); whether it differs
    // from map_ by more than regrid_tol; and the switch: interpolate *states (5th-order Lagrange
    // in xi) onto the new grid and call resolve(), restoring grid and states if it fails.
    RadialMap wanted_map(const Vec& x) const;
    bool map_drifted(const RadialMap& want) const;
    bool regrid(const RadialMap& want, const std::vector<Vec*>& states, const std::function<bool()>& resolve);
    Vec interpolate_state(const Vec& x, const RadialMap& old_map) const;
    bool load_init_file();  // init_file: interpolate a bs2d.dat profile onto the current grid
    bool init_map_valid_ = false;  // init_file header held a radial map (init_map_)
    RadialMap init_map_;

    void write_profile(const std::string& path, const Vec& x) const;
    void append_family(const std::string& path, const Vec& x, bool header) const;
};

#endif // ROTATING_SOLVER_H_

#ifndef SOLVER_H_
#define SOLVER_H_

#include <vector>
#include <string>
#include <utility>
#include "bs_params.h"

struct SolverState {
    std::vector<double> rbar;
    std::vector<double> r;

    // Spherical seed fields (m=0 baseline)
    std::vector<double> phi;
    std::vector<double> metric_f;
    std::vector<double> metric_l;

    // Auxiliary fields from Appendix A workflow
    std::vector<double> omega_s;
    std::vector<double> rho;
};

class BosonStarSolver {
public:
    explicit BosonStarSolver(const BSParams& params);

    void initialize_seed();
    bool run_relaxation_skeleton();
    void print_run_summary() const;

private:
    using Vec = std::vector<double>;

    struct Residuals {
        std::vector<double> f;
        std::vector<double> l;
        std::vector<double> phi;
        std::vector<double> omega;
        std::vector<double> rho;
        // Per-row sums of |terms| (round-off scale of each residual entry)
        std::vector<double> sf, sl, sphi, somega, srho;
    };

    BSParams p_;
    double current_kappa_ = 0.0;
    bool lapse_mode_active_ = false;
    bool seeded_from_qball_ = false;
    bool arclength_active_ = false;
    std::vector<double> arc_tangent_; // unit secant through the last two solutions (packed state)
    std::vector<double> arc_pred_;    // extrapolated predictor x_cur + ds * t
    double lapse_ratio_target_ = 0.0;
    SolverState state_;
    double last_residual_norm_ = 0.0;
    int last_iterations_ = 0;
    int last_iters_to_floor_ = 0;

    void build_grid();
    void apply_regular_seed_profile();

    // Potential helpers
    double potential_U(double phi) const;
    double dU_dphi(double phi) const;

    // Derivatives in compactified coordinate rbar
    double d1_rbar(const std::vector<double>& u, int i, double h) const;
    double d2_rbar(const std::vector<double>& u, int i, double h) const;
    // Same stencils with |coefficients| * |u| (round-off scale of the derivative)
    double d1_rbar_abs(const std::vector<double>& u, int i, double h) const;
    double d2_rbar_abs(const std::vector<double>& u, int i, double h) const;
    double dU_dphi_abs_terms(double phi) const;

    // Convert rbar-derivatives to physical-r derivatives (r = rbar/(1-rbar))
    double d1_r_from_rbar(double u_b, double rb) const;
    double d2_r_from_rbar(double u_b, double u_bb, double rb) const;

    Residuals compute_residuals() const;
    void pack_residual_vector(const Residuals& R, Vec& r) const;
    void pack_state_vector(Vec& x) const;
    void unpack_state_vector(const Vec& x);

    size_t active_field_count() const;
    size_t state_vector_size() const;

    double max_abs_residual(const Residuals& R) const;
    double inf_norm(const Vec& v) const;
    double l2_norm(const Vec& v) const;

    void jacobian_vector_product_fd(
        const Vec& x_base,
        const Vec& r_base,
        const Vec& v,
        Vec& jv);

    // Assemble full Jacobian (dense, row-major) at state x into J (size n*n).
    // Returns true on success. J is resized to n*n where n = state_vector_size().
    bool assemble_jacobian(const Vec& x_base, std::vector<double>& J);

    // Set state_ from a packed vector without clamping (for finite differences).
    void assign_state_raw(const Vec& x);

    // Colored central-difference Jacobian (band +-2 grid points, interleaved field
    // ordering) solved with banded LU; out-of-band couplings via Woodbury.
    bool assemble_and_solve_banded(const Vec& x_base, const Vec& rhs, Vec& delta);

    bool gmres_solve(
        const Vec& x_base,
        const Vec& r_base,
        const Vec& rhs,
        Vec& delta);

    bool apply_newton_step_with_line_search(const Vec& x, const Vec& delta, double current_norm);
    double apply_residual_smoother_step(const Residuals& R, double alpha);

    bool solve_current_kappa_stage();
    // Final check on a converged Newton state (rejects the trivial phi = 0 vacuum).
    bool accept_converged_state(const char* criterion) const;
    void pack_scale_vector(const Residuals& R, Vec& s) const;
    // |r_i| <= rel * scale_i + abs_tol_residual for all rows
    bool residual_within(const Vec& r, const Vec& scale, double rel) const;
    bool bootstrap_from_qball_shooting();
    bool potential_supports_qball_bootstrap() const;
    
    // Write solved boson-star profile (current state) to bs.dat.
    // Columns: r phi metric_f metric_l
    bool write_bs_dat();

    // Append or write a full model block to a families file. If overwrite is true,
    // the file is overwritten and a header is written. Otherwise the model is
    // appended as a new block. Each line has: kappa omega r phi metric_f metric_l rho
    bool append_family_model(const std::string& filename, bool overwrite, double omega);

    struct QBallProfile {
        std::vector<double> r;
        std::vector<double> phi;
        std::vector<double> dphi;
    };

    std::pair<double, double> qball_rhs(double r, double phi, double dphi, double omega) const;
    bool integrate_qball_profile(double phi0, double omega, QBallProfile& prof, double& phi_end, bool& crossed_zero) const;
    bool find_qball_central_amplitude(double omega, double& phi0_best, QBallProfile& best_profile) const;
    double interp_linear(const std::vector<double>& x, const std::vector<double>& y, double xi) const;
    bool apply_exponential_tail_to_qball_profile(QBallProfile& prof, double omega) const;
    bool write_qball_dat(const QBallProfile& prof, double omega) const;

    void enforce_boundary_values();

    // Compute the current global charge from the solver state (asymptotic)
    double compute_current_charge() const;

    // ADM mass from the exterior isotropic-Schwarzschild form of f at r = r[N-2], in units
    // G = kappa / (8 pi); the Komar integral at kappa = 0
    double compute_current_mass() const;
    // Volume integrals (cross-checks of the asymptotic M and the rho-based Q)
    double compute_komar_mass() const;
    double compute_charge_integral() const;
    double radial_integral(const std::vector<double>& integrand_r) const;

    bool is_charge_continuation() const;
    bool is_omega_continuation() const;
    bool is_lapse_continuation() const;
    bool is_arclength_continuation() const;
    // omega_s is part of the Newton state (charge, lapse, arclength modes)
    bool omega_is_unknown() const;

    // Pseudo-arclength continuation along the solution curve (see .cpp)
    bool run_arclength_continuation();
    double arclength_residual(double* scale = nullptr) const;
};

#endif // SOLVER_H_

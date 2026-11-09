#include "params_io.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mathutils.h"

BSParams read_params(const std::string& file_path) {
    std::ifstream in(file_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open parameter file: " + file_path);
    }

    BSParams p;
    std::string line;

    while (std::getline(in, line)) {
        fill_parameter(line, std::string("quiet = "), p.quiet, true);

        fill_parameter(line, std::string("sequence_mode = "), p.sequence_mode, true);
        fill_parameter(line, std::string("continuation_mode = "), p.continuation_mode, true);

        fill_parameter(line, std::string("dirac_dissipation = "), p.dirac_dissipation, true);
        fill_parameter(line, std::string("dirac_origin_zero = "), p.dirac_origin_zero, true);
        fill_parameter(line, std::string("dirac_parity = "), p.dirac_parity, true);
        fill_parameter(line, std::string("matter = "), p.matter, true);
        fill_parameter(line, std::string("potential = "), p.potential, true);
        fill_parameter(line, std::string("mu = "), p.mu, true);
        fill_parameter(line, std::string("kappa = "), p.kappa, true);
        fill_parameter(line, std::string("sigma = "), p.sigma, true);
        fill_parameter(line, std::string("kkls_kappa = "), p.kkls_kappa, true);
        fill_parameter(line, std::string("f_axion = "), p.f_axion, true);
        fill_parameter(line, std::string("lambda_rep = "), p.lambda_rep, true);
        fill_parameter(line, std::string("kleihaus_lambda = "), p.kleihaus_lambda, true);
        fill_parameter(line, std::string("kleihaus_a = "), p.kleihaus_a, true);
        fill_parameter(line, std::string("kleihaus_b = "), p.kleihaus_b, true);

        fill_parameter(line, std::string("omega_target = "), p.omega_target, true);
        fill_parameter(line, std::string("charge_target = "), p.charge_target, true);

        fill_parameter(line, std::string("Nr = "), p.Nr, true);
        fill_parameter(line, std::string("rbar_min = "), p.rbar_min, true);
        fill_parameter(line, std::string("rbar_max = "), p.rbar_max, true);
        fill_parameter(line, std::string("fd_order = "), p.fd_order, true);
        fill_parameter(line, std::string("max_newton_iters = "), p.max_newton_iters, true);
        fill_parameter(line, std::string("abs_tol_residual = "), p.abs_tol_residual, true);
        fill_parameter(line, std::string("abs_tol_update = "), p.abs_tol_update, true);
    fill_parameter(line, std::string("linear_tol = "), p.linear_tol, true);
        fill_parameter(line, std::string("line_search_min = "), p.line_search_min, true);
    fill_parameter(line, std::string("max_line_search_steps = "), p.max_line_search_steps, true);
    fill_parameter(line, std::string("armijo_c = "), p.armijo_c, true);
    fill_parameter(line, std::string("predictor_smooth_iters = "), p.predictor_smooth_iters, true);
    fill_parameter(line, std::string("gmres_restart = "), p.gmres_restart, true);
    fill_parameter(line, std::string("gmres_max_iters = "), p.gmres_max_iters, true);
    fill_parameter(line, std::string("jacobian_fd_eps = "), p.jacobian_fd_eps, true);
    fill_parameter(line, std::string("rel_tol_residual = "), p.rel_tol_residual, true);
    fill_parameter(line, std::string("stall_rel_tol = "), p.stall_rel_tol, true);
    fill_parameter(line, std::string("trivial_phi_tol = "), p.trivial_phi_tol, true);
        fill_parameter(line, std::string("linear_solver = "), p.linear_solver, true);

        fill_parameter(line, std::string("delta_kappa = "), p.delta_kappa, true);
        fill_parameter(line, std::string("delta_omega = "), p.delta_omega, true);
        fill_parameter(line, std::string("delta_lapse = "), p.delta_lapse, true);
        fill_parameter(line, std::string("delta_charge = "), p.delta_charge, true);
        fill_parameter(line, std::string("lapse_target = "), p.lapse_target, true);
    fill_parameter(line, std::string("kappa_target = "), p.kappa_target, true);
    fill_parameter(line, std::string("arclength_max_steps = "), p.arclength_max_steps, true);
    fill_parameter(line, std::string("arclength_ds_max = "), p.arclength_ds_max, true);
    fill_parameter(line, std::string("arclength_min_cos = "), p.arclength_min_cos, true);
    fill_parameter(line, std::string("arclength_omega_min = "), p.arclength_omega_min, true);
    fill_parameter(line, std::string("arclength_omega_max = "), p.arclength_omega_max, true);
    fill_parameter(line, std::string("core_resolution_tol = "), p.core_resolution_tol, true);
    fill_parameter(line, std::string("wall_resolution_tol = "), p.wall_resolution_tol, true);
    fill_parameter(line, std::string("first_law_tol = "), p.first_law_tol, true);
    fill_parameter(line, std::string("first_law_retries = "), p.first_law_retries, true);
    fill_parameter(line, std::string("write_bsdata_per_step = "), p.write_bsdata_per_step, true);

    fill_parameter(line, std::string("use_qball_bootstrap = "), p.use_qball_bootstrap, true);
    fill_parameter(line, std::string("qball_rmax = "), p.qball_rmax, true);
    fill_parameter(line, std::string("qball_dr = "), p.qball_dr, true);
    fill_parameter(line, std::string("qball_phi0_min = "), p.qball_phi0_min, true);
    fill_parameter(line, std::string("qball_phi0_max = "), p.qball_phi0_max, true);
    fill_parameter(line, std::string("qball_scan_points = "), p.qball_scan_points, true);
    fill_parameter(line, std::string("r_match_fac = "), p.r_match_fac, true);

        fill_parameter(line, std::string("Ntheta = "), p.Ntheta, true);
        fill_parameter(line, std::string("m_azimuthal = "), p.m_azimuthal, true);
        fill_parameter(line, std::string("rot_seed_amp = "), p.rot_seed_amp, true);
        fill_parameter(line, std::string("seed_file = "), p.seed_file, true);
        fill_parameter(line, std::string("init_file = "), p.init_file, true);
        fill_parameter(line, std::string("init_file_prev = "), p.init_file_prev, true);
        fill_parameter(line, std::string("rot_seed_width = "), p.rot_seed_width, true);
        fill_parameter(line, std::string("single_model_phi_max = "), p.single_model_phi_max, true);
        fill_parameter(line, std::string("rot_omega_stop = "), p.rot_omega_stop, true);
        fill_parameter(line, std::string("rot_max_steps = "), p.rot_max_steps, true);
        fill_parameter(line, std::string("radial_grid = "), p.radial_grid, true);
        fill_parameter(line, std::string("light_rings = "), p.light_rings, true);
        fill_parameter(line, std::string("ergoregion = "), p.ergoregion, true);
        fill_parameter(line, std::string("radial_scale = "), p.radial_scale, true);
        fill_parameter(line, std::string("radial_cluster_fraction = "), p.radial_cluster_fraction, true);
        fill_parameter(line, std::string("radial_scale_ratio = "), p.radial_scale_ratio, true);
        fill_parameter(line, std::string("regrid_tol = "), p.regrid_tol, true);
        fill_parameter(line, std::string("jacobian_reuse = "), p.jacobian_reuse, true);
        fill_parameter(line, std::string("chord_rate = "), p.chord_rate, true);
        fill_parameter(line, std::string("lu_pivot_threshold = "), p.lu_pivot_threshold, true);
        fill_parameter(line, std::string("arclength_predictor = "), p.arclength_predictor, true);
        fill_parameter(line, std::string("newton_linear = "), p.newton_linear, true);
        fill_parameter(line, std::string("gmres_lu_max_iters = "), p.gmres_lu_max_iters, true);
        fill_parameter(line, std::string("linear_preconditioner = "), p.linear_preconditioner, true);
        fill_parameter(line, std::string("ilut_fill = "), p.ilut_fill, true);
        fill_parameter(line, std::string("ilut_droptol = "), p.ilut_droptol, true);
        fill_parameter(line, std::string("ilut_float = "), p.ilut_float, true);
        fill_parameter(line, std::string("ilut_lu_fallback = "), p.ilut_lu_fallback, true);
        fill_parameter(line, std::string("gmres_lu_tol = "), p.gmres_lu_tol, true);
        fill_parameter(line, std::string("output_prefix = "), p.output_prefix, true);
    }

    if (p.Nr < 5) {
        throw std::runtime_error("Nr must be at least 5 for high-order stencils.");
    }
    if (p.rbar_max <= p.rbar_min) {
        throw std::runtime_error("Require rbar_max > rbar_min.");
    }

    return p;
}

void print_params_summary(const BSParams& p) {
    std::cout << "\n==== Boson Star Solver Configuration ====\n";
    std::cout << "mode                : " << p.sequence_mode << "\n";
    std::cout << "continuation        : " << p.continuation_mode << "\n";
    std::cout << "potential           : " << p.potential << "\n";
    std::cout << "mu                  : " << p.mu << "\n";
    std::cout << "kappa               : " << p.kappa << "\n";
    std::cout << "omega_target        : " << p.omega_target << "\n";
    std::cout << "charge_target       : " << p.charge_target << "\n";
    std::cout << "lapse_target        : " << p.lapse_target << "\n";
    std::cout << "delta_lapse         : " << p.delta_lapse << "\n";
    std::cout << "Nr                  : " << p.Nr << "\n";
    std::cout << "rbar range          : [" << p.rbar_min << ", " << p.rbar_max << "]\n";
    std::cout << "max_newton_iters    : " << p.max_newton_iters << "\n";
    std::cout << "abs_tol_residual    : " << p.abs_tol_residual << "\n";
    std::cout << "abs_tol_update      : " << p.abs_tol_update << "\n";
    std::cout << "linear_tol          : " << p.linear_tol << "\n";
    std::cout << "gmres_restart       : " << p.gmres_restart << "\n";
    std::cout << "gmres_max_iters     : " << p.gmres_max_iters << "\n";
    std::cout << "jacobian_fd_eps     : " << p.jacobian_fd_eps << "\n";
    std::cout << "linear_solver       : " << p.linear_solver << "\n";
    std::cout << "use_qball_bootstrap : " << p.use_qball_bootstrap << "\n";
    std::cout << "kappa_target        : " << p.kappa_target << "\n";
    std::cout << "write_bsdata_per_step: " << p.write_bsdata_per_step << "\n";
    std::cout << "qball_rmax          : " << p.qball_rmax << "\n";
    std::cout << "qball_dr            : " << p.qball_dr << "\n";
    std::cout << "r_match_fac         : " << p.r_match_fac << "\n";
    std::cout << "==========================================\n\n";
}

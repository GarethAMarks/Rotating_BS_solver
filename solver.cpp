#include "solver.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include "mathutils.h"
#include "linear_solvers.h"
#include "potentials.h"

namespace {
constexpr double kPi = 3.1415926535897932384626433832795;
}

BosonStarSolver::BosonStarSolver(const BSParams& params) : p_(params) {}

void BosonStarSolver::initialize_seed() {
    build_grid();
    current_kappa_ = 0.0;

    bool seeded = false;
    if (p_.use_qball_bootstrap && potential_supports_qball_bootstrap()) {
        seeded = bootstrap_from_qball_shooting();
        if (!seeded) {
            std::cout << "[warn] Q-ball shooting bootstrap failed; using analytic regular seed.\n";
        }
    } else if (p_.use_qball_bootstrap && !potential_supports_qball_bootstrap()) {
        std::cout << "[info] Q-ball bootstrap skipped: selected potential has no flat-space Q-ball branch.\n";
    }

    if (!seeded) {
        apply_regular_seed_profile();
    }
    seeded_from_qball_ = seeded;
}

bool BosonStarSolver::run_relaxation_skeleton() {
    std::cout << "[info] Starting staged solve: Q-ball seed -> kappa continuation -> Newton-Krylov refinement.\n";

    // Lapse constraints should only be active while the dedicated lapse
    // continuation loop is running, not during kappa continuation.
    lapse_mode_active_ = false;

    // Use explicit kappa_target if provided; otherwise fall back to legacy p_.kappa
    const double target_kappa = std::max(0.0, p_.kappa_target > 0.0 ? p_.kappa_target : p_.kappa);
    const double dk = std::max(1e-8, p_.delta_kappa);

    if (target_kappa <= 0.0) {
        // Flat space (Q-balls): solve at kappa = 0, then continue the family below.
        current_kappa_ = 0.0;
        if (!solve_current_kappa_stage()) {
            write_bs_dat();
            return false;
        }
    }

    // Without a Q-ball seed there is nothing to continue from at kappa = 0: for mini
    // and repulsive potentials the kappa = 0 problem is linear in phi and only phi = 0
    // solves it. Solve directly at the target kappa from the analytic seed instead.
    if (!seeded_from_qball_ && dk < target_kappa) {
        std::cout << "[info] No Q-ball seed; solving directly at kappa = " << target_kappa << ".\n";
        current_kappa_ = target_kappa - dk;
    }

    int stage = 0;
        // Continue until the target kappa is reached (inclusive). No artificial stage cap.
        while (current_kappa_ < target_kappa - 1e-14) {
        current_kappa_ = std::min(target_kappa, current_kappa_ + dk);
        ++stage;
        std::cout << "[info] Continuation stage " << stage << ": kappa = " << current_kappa_ << "\n";

        const bool ok = solve_current_kappa_stage();
        if (!ok) {
            std::cout << "[warn] Stage failed at kappa = " << current_kappa_ << "\n";
            // Provide extra diagnostics to help debug kappa-stage failures.
            const int Ndbg = p_.Nr;
            const int iAsymDbg = std::max(0, Ndbg - 2);
            const Residuals Rfail = compute_residuals();
            auto find_max = [&](const std::vector<double>& v){
                double mv = 0.0; int idx = 0;
                for (int ii = 0; ii < Ndbg; ++ii) {
                    const double av = std::abs(v[static_cast<size_t>(ii)]);
                    if (av > mv) { mv = av; idx = ii; }
                }
                return std::make_pair(mv, idx);
            };
            auto [mf, ifi] = find_max(Rfail.f);
            auto [ml, ili] = find_max(Rfail.l);
            auto [mp, ipi] = find_max(Rfail.phi);
            auto [mr, iri] = find_max(Rfail.rho);
            auto [mw, iwi] = find_max(Rfail.omega);

            std::cout << std::scientific << std::setprecision(6);
            std::cout << "[diag] residual maxes -- f:" << mf << "@" << ifi
                      << " l:" << ml << "@" << ili
                      << " phi:" << mp << "@" << ipi
                      << " rho:" << mr << "@" << iri
                      << " omega:" << mw << "@" << iwi << "\n";

            std::cout << "[diag] R_f(inf)=" << Rfail.f[static_cast<size_t>(Ndbg-1)]
                      << " R_l(inf)=" << Rfail.l[static_cast<size_t>(Ndbg-1)]
                      << " R_phi(inf)=" << Rfail.phi[static_cast<size_t>(Ndbg-1)]
                      << " R_rho(inf)=" << Rfail.rho[static_cast<size_t>(Ndbg-1)]
                      << " R_omega(inf)=" << Rfail.omega[static_cast<size_t>(Ndbg-1)] << "\n";

            std::cout << "[diag] R_f(0)=" << Rfail.f[0]
                      << " R_l(0)=" << Rfail.l[0]
                      << " R_phi(0)=" << Rfail.phi[0]
                      << " R_rho(0)=" << Rfail.rho[0]
                      << " R_omega(0)=" << Rfail.omega[0] << "\n";

            // Print current state indicators
            std::cout << "[diag] state metric_f(0)=" << state_.metric_f[0]
                      << " sampled omega_asym=" << state_.omega_s[static_cast<size_t>(iAsymDbg)] << "\n";
            std::cout << "[diag] omega_s[0..4] ";
            for (int ii = 0; ii < std::min(5, Ndbg); ++ii) std::cout << state_.omega_s[static_cast<size_t>(ii)] << " ";
            std::cout << "\n";

            // Pack residual vector norms
            Vec rv; pack_residual_vector(Rfail, rv);
            double infn = inf_norm(rv);
            double l2n = l2_norm(rv);
            std::cout << "[diag] residual norms: inf=" << infn << " l2=" << l2n << " last_residual_inf=" << last_residual_norm_ << " last_iters=" << last_iterations_ << "\n";

            // Write current best state even on failure so user can inspect.
            if (write_bs_dat()) {
                std::cout << "[info] Wrote boson-star profile (partial) to bs.dat\n";
            } else {
                std::cout << "[warn] Failed to write bs.dat\n";
            }
            return false;
        } else {
            // Successful stage: write per-stage diagnostics and optional branch table
            if (p_.write_bsdata_per_step) {
                // Compute a small set of diagnostics for the branch table
                double max_phi = 0.0;
                for (double v : state_.phi) max_phi = std::max(max_phi, std::abs(v));

                // If first stage, overwrite and write header; otherwise append
                std::ofstream bout;
                if (stage == 1) {
                    bout.open("initial_models.dat");
                    bout << "# kappa omega_target max_phi mass charge residual_inf\n";
                } else {
                    bout.open("initial_models.dat", std::ios::app);
                }
                if (bout.is_open()) {
                    const double mass = compute_current_mass();
                    const double charge = compute_current_charge();

                    bout << std::scientific << std::setprecision(12)
                         << current_kappa_ << ' ' << p_.omega_target << ' ' << max_phi << ' ' << mass << ' ' << charge << ' ' << last_residual_norm_ << '\n';
                }
            }

            // Also write the full profile snapshot for this stage (overwrite bs.dat)
            if (write_bs_dat()) {
                std::cout << "[info] Wrote boson-star profile to bs.dat\n";
            } else {
                std::cout << "[warn] Failed to write bs.dat\n";
            }
        }
    }

        // Finished continuation loop; target kappa reached (or loop terminated).

    if (write_bs_dat()) {
        std::cout << "[info] Wrote boson-star profile to bs.dat\n";
    } else {
        std::cout << "[warn] Failed to write bs.dat\n";
    }

    // Write the final model produced by the kappa continuation as the first
    // entry in the families file (overwrite any existing file).
    if (!append_family_model("families.dat", true, p_.omega_target)) {
        std::cout << "[warn] Failed to write families.dat (initial model)\n";
    } else {
        std::cout << "[info] Wrote families initial model to families.dat\n";
    }

    // Seed omega/charge history with the current solved model so tangent
    // information is available once we take another omega step.
    if (p_.continuation_mode != "lapse" && p_.delta_omega > 0.0) {
        const double init_charge = compute_current_charge();
        // seed history vectors with the current solved model so tangent
        // information is available once we take another omega step.
        // (these will be pushed into the actual vectors below)
        (void)init_charge; // placeholder to avoid unused-warning if not used later
    }

    if (p_.continuation_mode == "arclength") {
        return run_arclength_continuation();
    }

    // Now, keeping kappa fixed at current_kappa_, decrement omega and continue
    // using the last solved boson-star as the seed. Stop when relaxation fails
    // or omega drops below zero.
    // If user requested lapse continuation, run that mode instead of omega.
    if (p_.continuation_mode == "lapse" && p_.delta_lapse > 0.0) {
        double lapse = std::sqrt(std::max(1e-16, state_.metric_f[0]));
        int lapse_stage = 0;
        std::vector<double> omega_history;
        std::vector<double> lapse_history;
        std::vector<double> ratio_history;

    // Enter explicit lapse-continuation phase.
        lapse_mode_active_ = true;
        p_.lapse_target = lapse;

        // seed histories with current solved model
        {
            const int N = p_.Nr;
            const int iAsym = std::max(0, N - 2);
            const double seed_omega = state_.omega_s[static_cast<size_t>(iAsym)];
            omega_history.push_back(seed_omega);
            lapse_history.push_back(lapse);
        }

        // In lapse mode, use the coupled continuation variable
        // ratio = omega_asym / lapse_center, and step this ratio upward.
        // For each candidate ratio, derive lapse_target = omega_seed / ratio.
        const double init_rstep = std::max(1e-12, p_.delta_lapse);
        double ratio = 0.0;
        if (!omega_history.empty() && lapse > 1e-16) {
            ratio = omega_history.back() / lapse;
        }
        lapse_ratio_target_ = ratio;
        ratio_history.push_back(ratio);

        while (lapse > 0.0) {
            bool stage_ok = false;
            double try_step = init_rstep;
            double min_step_allowed = std::max(1e-12, init_rstep * 1e-8);

            while (try_step >= min_step_allowed) {
                const int Nseed = p_.Nr;
                const int iAsymSeed = std::max(0, Nseed - 2);
                const double seed_omega = (!omega_history.empty())
                    ? omega_history.back()
                    : state_.omega_s[static_cast<size_t>(iAsymSeed)];

                if (std::abs(seed_omega) < 1e-14) {
                    std::cout << "[warn] Seed omega too small in ratio-based lapse continuation; aborting.\n";
                    break;
                }

                bool solved_this_step = false;
                // Try both ratio directions for robustness in lapse mode.
                for (int sgn_i = 0; sgn_i < 2 && !solved_this_step; ++sgn_i) {
                    const double dir = (sgn_i == 0) ? +1.0 : -1.0;
                    const double dq_try = dir * try_step;
                    const double cand_ratio = ratio + dq_try;
                    if (cand_ratio <= 1e-16) continue;

                    // Predict omega for this continuation step (secant predictor)
                    // so ratio-stepping doesn't algebraically pin omega to the
                    // previous seed via lapse_target = seed_omega / cand_ratio.
                    double omega_pred = seed_omega;
                    if (omega_history.size() >= 2 && ratio_history.size() >= 2) {
                        const size_t m = omega_history.size();
                        const double dw = omega_history[m - 1] - omega_history[m - 2];
                        const double dq = ratio_history[m - 1] - ratio_history[m - 2];
                        if (std::abs(dq) > 1e-14) {
                            const double slope_dw_dq = dw / dq;
                            omega_pred = omega_history.back() + slope_dw_dq * dq_try;
                        } else {
                            omega_pred = omega_history.back() + dir * dw;
                        }
                    }
                    if (!std::isfinite(omega_pred) || std::abs(omega_pred) < 1e-14) {
                        omega_pred = seed_omega;
                    }

                    const double cand = omega_pred / cand_ratio;
                    if (cand <= 0.0) {
                        std::cout << "[warn] Candidate lapse became non-positive (cand=" << cand
                                  << ") for ratio=" << cand_ratio << "; trying alternate direction/step.\n";
                        continue;
                    }

                    // set target lapse (we store target as sqrt(f0))
                    p_.lapse_target = cand;
                    lapse_ratio_target_ = cand_ratio;

                    // Seed the solver state with a consistent guess informed by the
                    // previous solved model: set the origin metric to the target
                    // value and seed omega with the continuation predictor.
                    const double f0_target = p_.lapse_target * p_.lapse_target;
                    if (Nseed > 0) state_.metric_f[0] = std::max(1e-12, f0_target);
                    std::fill(state_.omega_s.begin(), state_.omega_s.end(), omega_pred);

                    std::cout << "[info] Trying ratio-based lapse continuation: target_ratio="
                              << cand_ratio << " target_lapse=" << cand
                              << " omega_pred=" << omega_pred
                              << " rstep=" << try_step
                              << " dir=" << ((dir > 0.0) ? "+" : "-") << "\n";

                    // Ensure infinity/outer constraints are applied after seeding.
                    enforce_boundary_values();
                    // predictor smoothing
                    const double alpha_pred = 1e-2;
                    double maxupd = 0.0;
                    for (int psi = 0; psi < p_.predictor_smooth_iters; ++psi) {
                        const Residuals Rpred = compute_residuals();
                        const double this_upd = apply_residual_smoother_step(Rpred, alpha_pred);
                        maxupd = std::max(maxupd, this_upd);
                    }
                    if (maxupd > 0.0) std::cout << "[info] Applied predictor smoother (alpha=" << alpha_pred << ") max_update=" << maxupd << "\n";

                    const bool ok = solve_current_kappa_stage();
                    if (ok) {
                        const int N = p_.Nr;
                        const int iAsym = std::max(0, N - 2);
                        const double solved_omega = state_.omega_s[static_cast<size_t>(iAsym)];
                        // record diagnostics for tangent/arclength control
                        double lapse_c = std::sqrt(std::max(1e-16, state_.metric_f[0]));
                        omega_history.push_back(solved_omega);
                        lapse_history.push_back(lapse_c);
                        lapse = lapse_c;
                        ratio = (lapse_c > 1e-16) ? (solved_omega / lapse_c) : ratio;
                        ratio_history.push_back(ratio);
                        lapse_ratio_target_ = ratio;
                        ++lapse_stage;
                        stage_ok = true;
                        solved_this_step = true;
                        std::cout << "[info] Lapse continuation stage " << lapse_stage
                                  << ": lapse = " << lapse
                                  << " omega/lapse = " << ratio
                                  << " (rstep=" << try_step << ") succeeded\n";

                        if (!append_family_model("families.dat", false, solved_omega)) {
                            std::cout << "[warn] Failed to append model to families.dat at lapse=" << lapse << "\n";
                            stage_ok = false;
                        } else {
                            std::cout << "[info] Appended model at lapse = " << lapse << " to families.dat\n";
                        }

                        if (write_bs_dat()) std::cout << "[info] Wrote boson-star profile to bs.dat\n";
                        break;
                    }

                    std::cout << "[warn] Ratio/lapse candidate failed (ratio=" << cand_ratio
                              << ", lapse=" << cand << ", dir=" << ((dir > 0.0) ? "+" : "-")
                              << "); trying alternate direction/halved step\n";
                }

                if (stage_ok) break;
                try_step *= 0.5;
            }

            if (!stage_ok) {
                std::cout << "[warn] Lapse continuation could not advance from lapse=" << lapse << "; aborting lapse continuation.\n";
                break;
            }
        }

        // Leave lapse-continuation phase.
        lapse_mode_active_ = false;
    } 
    else if (p_.delta_omega > 0.0) {
    // (removed) unused adaptive-halving retry caps; omega halving handled inline

        double omega = p_.omega_target;
        int omega_stage = 0;
    // track a short history of computed charges and omegas to infer trend prior to failure
    std::vector<double> charge_history;
    std::vector<double> omega_history;
    std::vector<double> lapse_history;

    // Seed histories with the current solution so the tangent can be
    // computed after the first successful omega step. Use p_.omega_target
    // which holds the omega of the current solved model.
    {
        const double seed_charge = compute_current_charge();
        const double seed_lapse = std::sqrt(std::max(1e-16, state_.metric_f[0]));
        charge_history.push_back(seed_charge);
        omega_history.push_back(p_.omega_target);
        lapse_history.push_back(seed_lapse);
    }

    // Arclength target in the (omega, lapse_center) plane. We initialize
    // to the provided delta_omega and then set it to the observed arclength
    // after the first successful omega step for local scaling.
    const double init_step = p_.delta_omega;
    double ds_target = init_step;
    bool ds_target_set = false;

    double last_failed_omega_cand = -1.0;
    while (omega > 0.0) {
            // Compute adaptive omega step so that the arclength in the
            // (omega, lapse(0)) plane remains approximately ds_target.
            double step = init_step;
            if (omega_history.size() >= 2 && lapse_history.size() >= 2) {
                const size_t m = omega_history.size();
                const double w1 = omega_history[m - 2];
                const double w2 = omega_history[m - 1];
                const double l1 = lapse_history[m - 2];
                const double l2 = lapse_history[m - 1];
                const double dw = w2 - w1;
                const double dl = l2 - l1;
                // estimate local slope dl/dw
                double slope = 0.0;
                if (std::abs(dw) > 1e-16) slope = dl / dw;
                // choose step so that sqrt(dw^2 + (slope*dw)^2) = ds_target
                // => |dw| = ds_target / sqrt(1 + slope^2)
                const double denom = std::sqrt(1.0 + slope * slope);
                if (denom > 0.0) {
                    double proposed = ds_target / denom;
                    // guard proposed step to reasonable bounds
                    const double max_step = std::max(1e-6, init_step * 10.0);
                    const double min_step = std::max(1e-12, init_step * 1e-6);
                    if (proposed > max_step) proposed = max_step;
                    if (proposed < min_step) proposed = min_step;
                    step = proposed;
                }
            }
            bool stage_ok = false;

            {
                const double cand = omega - step;
                if (cand >= 0.0) {
                    // update target omega and use current state as seed
                    p_.omega_target = cand;
                    std::cout << "[info] Trying omega continuation: target=" << cand << " step=" << step << "\n";

                    // Predictor smoother
                    if (is_omega_continuation()) {
                        enforce_boundary_values();
                        const double alpha_pred = 1e-2;
                        double maxupd = 0.0;
                        for (int psi = 0; psi < p_.predictor_smooth_iters; ++psi) {
                            const Residuals Rpred = compute_residuals();
                            const double this_upd = apply_residual_smoother_step(Rpred, alpha_pred);
                            maxupd = std::max(maxupd, this_upd);
                        }
                        if (maxupd > 0.0) std::cout << "[info] Applied predictor smoother (alpha=" << alpha_pred << ") max_update=" << maxupd << "\n";
                    }

                    const bool ok = solve_current_kappa_stage();
                    if (ok) {
                        double current_charge = compute_current_charge();
                        // record diagnostics for tangent/arclength control
                        double lapse_c = std::sqrt(std::max(1e-16, state_.metric_f[0]));
                        charge_history.push_back(current_charge);
                        omega_history.push_back(cand);
                        lapse_history.push_back(lapse_c);
                        omega = cand;
                        ++omega_stage;
                        stage_ok = true;
                        std::cout << "[info] Omega continuation stage " << omega_stage << ": omega = " << omega << " (step=" << step << ") succeeded\n";

                        if (!append_family_model("families.dat", false, omega)) {
                            std::cout << "[warn] Failed to append model to families.dat at omega = " << omega << "\n";
                            stage_ok = false;
                        } else {
                            std::cout << "[info] Appended model at omega = " << omega << " to families.dat\n";
                        }

                        if (write_bs_dat()) std::cout << "[info] Wrote boson-star profile to bs.dat\n";
                        // After the first successful omega step, set ds_target to the
                        // observed arclength so subsequent steps aim to maintain it.
                        if (!ds_target_set && omega_history.size() >= 2 && lapse_history.size() >= 2) {
                            const size_t m = omega_history.size();
                            const double dw = omega_history[m - 1] - omega_history[m - 2];
                            const double dl = lapse_history[m - 1] - lapse_history[m - 2];
                            const double ds = std::sqrt(dw * dw + dl * dl);
                            if (ds > 0.0) {
                                ds_target = ds;
                                ds_target_set = true;
                                std::cout << "[info] Set target arclength ds_target=" << ds_target << " (based on first step)\n";
                            }
                        }
                    } else {
                        // failure: record the failed omega candidate, print diagnostics
                        last_failed_omega_cand = cand;
                        // fall through to mode-switch
                        std::cout << "[warn] Omega candidate " << cand << " failed; dumping residual diagnostics...\n";
                        const Residuals Rfail = compute_residuals();
                        const int N = p_.Nr;
                        auto find_max = [&](const std::vector<double>& v){
                            double mv = 0.0; int idx = 0;
                            for (int ii = 0; ii < N; ++ii) {
                                const double av = std::abs(v[static_cast<size_t>(ii)]);
                                if (av > mv) { mv = av; idx = ii; }
                            }
                            return std::make_pair(mv, idx);
                        };
                        auto [mf, ifi] = find_max(Rfail.f);
                        auto [ml, ili] = find_max(Rfail.l);
                        auto [mp, ipi] = find_max(Rfail.phi);
                        auto [mr, iri] = find_max(Rfail.rho);
                        auto [mw, iwi] = find_max(Rfail.omega);

                        std::cout << std::scientific << std::setprecision(6);
                        std::cout << "[diag] residual maxes -- f:" << mf << "@" << ifi
                                  << " l:" << ml << "@" << ili
                                  << " phi:" << mp << "@" << ipi
                                  << " rho:" << mr << "@" << iri
                                  << " omega:" << mw << "@" << iwi << "\n";

                        std::cout << "[diag] R_f(inf)=" << Rfail.f[static_cast<size_t>(N-1)]
                                  << " R_l(inf)=" << Rfail.l[static_cast<size_t>(N-1)]
                                  << " R_phi(inf)=" << Rfail.phi[static_cast<size_t>(N-1)]
                                  << " R_rho(inf)=" << Rfail.rho[static_cast<size_t>(N-1)]
                                  << " R_omega(inf)=" << Rfail.omega[static_cast<size_t>(N-1)] << "\n";

                        std::cout << "[diag] R_f(0)=" << Rfail.f[0]
                                  << " R_l(0)=" << Rfail.l[0]
                                  << " R_phi(0)=" << Rfail.phi[0]
                                  << " R_rho(0)=" << Rfail.rho[0]
                                  << " R_omega(0)=" << Rfail.omega[0] << "\n";

                        Vec rv; pack_residual_vector(Rfail, rv);
                        double infn = inf_norm(rv);
                        double l2n = l2_norm(rv);
                        std::cout << "[diag] residual norms: inf=" << infn << " l2=" << l2n << "\n";
                    }
                }
            }

            if (!stage_ok) {
                std::cout << "[warn] Omega continuation aborted after retries; last omega=" << omega << "\n";

                // Attempt mode switch: try charge continuation if currently in omega
                if (is_omega_continuation()) {
                    std::cout << "[info] Switching to charge continuation mode after omega failure.\n";
                    // Infer direction of charge change prior to failure using the
                    // tangent in the (omega, Q) plane if possible. This helps follow
                    // the spiral instead of backtracking.
                    int charge_dir = 0; // +1 => increase, -1 => decrease
                    if (omega_history.size() >= 2 && charge_history.size() >= 2) {
                        const size_t n = charge_history.size();
                        const double dq = charge_history[n - 1] - charge_history[n - 2];
                        const double dom = omega_history[n - 1] - omega_history[n - 2];
                        // Tangent in (omega,Q) is (dom, dq). Prefer the sign of dq
                        // to decide whether to increase or decrease Q when switching
                        // to charge continuation. If dq is effectively zero, fall
                        // back to the previous heuristic.
                        if (std::abs(dq) > 1e-14) {
                            charge_dir = (dq > 0.0) ? +1 : -1;
                            std::cout << "[info] Tangent-based charge_dir = " << charge_dir << " (dq=" << dq << ", dom=" << dom << ")\n";
                        }
                    }
                    if (charge_dir == 0 && charge_history.size() >= 2) {
                        const double last = charge_history.back();
                        const double prev = charge_history[charge_history.size() - 2];
                        charge_dir = (last > prev) ? +1 : (last < prev ? -1 : 0);
                    }
                    if (charge_dir == 0) {
                        std::cout << "[warn] Unable to infer charge trend from history; defaulting to decrementing charge.\n";
                        charge_dir = -1;
                    }

                    // Switch continuation mode to charge
                    p_.continuation_mode = "charge";

                    // Start persistent charge-continuation: keep stepping charge until it fails.
                    const double init_cstep_abs = std::abs(p_.delta_charge);
                    double ccharge = compute_current_charge();
                    // Use a reduced initial step for the first few attempts after switching
                    int small_steps_remaining = 3;

                    // Try advancing charge repeatedly; on failure we'll try the opposite direction
                    for (;;) {
                        // Use a smaller initial cstep for the first few iterations
                        double base_cstep = init_cstep_abs;
                        if (small_steps_remaining > 0) base_cstep *= 0.1;

                        // Try a single charge candidate with adaptive halving (local retries)
                        double cstep = base_cstep;
                        bool charge_stage_ok = false;
                        double cand_charge = ccharge;

                        while (cstep >= 1e-12) {
                            cand_charge = ccharge + static_cast<double>(charge_dir) * cstep;
                            std::cout << "[info] Trying charge continuation: target_charge=" << cand_charge << " step=" << cstep << "\n";

                            // set target and synchronize
                            p_.charge_target = cand_charge;
                            enforce_boundary_values();
                            const double alpha_pred = 1e-2;
                            double maxupd = 0.0;
                            for (int psi = 0; psi < p_.predictor_smooth_iters; ++psi) {
                                const Residuals Rpred = compute_residuals();
                                const double this_upd = apply_residual_smoother_step(Rpred, alpha_pred);
                                maxupd = std::max(maxupd, this_upd);
                            }
                            if (maxupd > 0.0) std::cout << "[info] Applied predictor smoother (alpha=" << alpha_pred << ") max_update=" << maxupd << "\n";

                            const bool cok = solve_current_kappa_stage();
                            if (cok) {
                                // success: append model and commit charge step
                                const int N = p_.Nr;
                                const int iAsym = std::max(0, N - 2);
                                const double solved_omega = state_.omega_s[static_cast<size_t>(iAsym)];
                                if (!append_family_model("families.dat", false, solved_omega)) {
                                    std::cout << "[warn] Failed to append model to families.dat during charge continuation.\n";
                                    // treat as failure to keep logic simple
                                    charge_stage_ok = false;
                                    break;
                                }
                                // also write a snapshot
                                if (write_bs_dat()) std::cout << "[info] Wrote boson-star profile to bs.dat\n";
                                std::cout << "[info] Charge continuation succeeded (charge=" << cand_charge << ")\n";
                                // commit
                                ccharge = cand_charge;
                                charge_stage_ok = true;
                                // consume one small-step quota if applicable
                                if (small_steps_remaining > 0) --small_steps_remaining;
                                break;
                            }

                            // On failure, first try the opposite direction at the same step
                            const int alt_dir_try = -charge_dir;
                            const double alt_cand = ccharge + static_cast<double>(alt_dir_try) * cstep;
                            std::cout << "[info] Primary charge candidate failed; trying opposite direction target_charge=" << alt_cand << " at same step=" << cstep << "\n";
                            p_.charge_target = alt_cand;
                            enforce_boundary_values();
                            double maxupd_alt = 0.0;
                            for (int psi = 0; psi < p_.predictor_smooth_iters; ++psi) {
                                const Residuals Rpred = compute_residuals();
                                const double this_upd = apply_residual_smoother_step(Rpred, alpha_pred);
                                maxupd_alt = std::max(maxupd_alt, this_upd);
                            }
                            if (maxupd_alt > 0.0) std::cout << "[info] Applied predictor smoother (alpha=" << alpha_pred << ") max_update=" << maxupd_alt << "\n";
                            const bool acok = solve_current_kappa_stage();
                            if (acok) {
                                const int N = p_.Nr;
                                const int iAsym = std::max(0, N - 2);
                                const double solved_omega = state_.omega_s[static_cast<size_t>(iAsym)];
                                if (!append_family_model("families.dat", false, solved_omega)) {
                                    std::cout << "[warn] Failed to append model to families.dat during alternate charge continuation.\n";
                                    charge_stage_ok = false;
                                } else {
                                    if (write_bs_dat()) std::cout << "[info] Wrote boson-star profile to bs.dat\n";
                                    std::cout << "[info] Alternate charge continuation succeeded (charge=" << alt_cand << ")\n";
                                    ccharge = alt_cand;
                                    // switch continuing direction to the alternate (decrease)
                                    charge_dir = alt_dir_try;
                                    charge_stage_ok = true;
                                    if (small_steps_remaining > 0) --small_steps_remaining;
                                }
                                break;
                            }

                            // Opposite-direction attempt also failed: fall back to halving and retry same direction
                            std::cout << "[warn] Both primary and opposite charge candidates failed at step=" << cstep << "; halving and retrying\n";
                            cstep *= 0.5;
                        }

                        if (!charge_stage_ok) {
                            // Charge continuation could not advance in the inferred direction.
                            // Try the opposite direction once (with halving retries) before
                            // giving up and retrying the failed omega candidate.
                            const int alt_dir = -charge_dir;
                            bool alt_ok = false;
                            double alt_cstep = init_cstep_abs;
                            while (alt_cstep >= 1e-12) {
                                const double alt_cand = ccharge + static_cast<double>(alt_dir) * alt_cstep;
                                std::cout << "[info] Trying alternate charge direction: target_charge=" << alt_cand << " step=" << alt_cstep << "\n";
                                p_.charge_target = alt_cand;
                                enforce_boundary_values();
                                const double alpha_pred = 1e-2;
                                double maxupd = 0.0;
                                for (int psi = 0; psi < p_.predictor_smooth_iters; ++psi) {
                                    const Residuals Rpred = compute_residuals();
                                    const double this_upd = apply_residual_smoother_step(Rpred, alpha_pred);
                                    maxupd = std::max(maxupd, this_upd);
                                }
                                if (maxupd > 0.0) std::cout << "[info] Applied predictor smoother (alpha=" << alpha_pred << ") max_update=" << maxupd << "\n";

                                const bool acok = solve_current_kappa_stage();
                                if (acok) {
                                    const int N = p_.Nr;
                                    const int iAsym = std::max(0, N - 2);
                                    const double solved_omega = state_.omega_s[static_cast<size_t>(iAsym)];
                                    if (!append_family_model("families.dat", false, solved_omega)) {
                                        std::cout << "[warn] Failed to append model to families.dat during alternate charge continuation.\n";
                                        alt_ok = false;
                                    } else {
                                        std::cout << "[info] Alternate charge continuation succeeded (charge=" << alt_cand << ")\n";
                                        ccharge = alt_cand;
                                        alt_ok = true;
                                    }
                                    break;
                                }

                                std::cout << "[warn] Alternate charge candidate " << alt_cand << " failed; halving step and retrying\n";
                                alt_cstep *= 0.5;
                            }

                            if (alt_ok) {
                                // alternate direction succeeded; continue persistent charge loop
                                continue;
                            }

                            // Neither direction could advance: retry the original failed omega candidate once.
                            std::cout << "[info] Charge continuation exhausted for charge=" << ccharge << "; will retry failed omega candidate " << last_failed_omega_cand << "\n";
                            p_.continuation_mode = "omega";
                            if (last_failed_omega_cand < 0.0) {
                                std::cout << "[error] No failed omega candidate recorded to retry; aborting.\n";
                                return false;
                            }

                            // Attempt the previously failed omega candidate once.
                            p_.omega_target = last_failed_omega_cand;
                            enforce_boundary_values();
                            // small predictor smoother for omega
                            if (is_omega_continuation()) {
                                const double alpha_pred = 1e-2;
                                double maxupd = 0.0;
                                for (int psi = 0; psi < p_.predictor_smooth_iters; ++psi) {
                                    const Residuals Rpred = compute_residuals();
                                    const double this_upd = apply_residual_smoother_step(Rpred, alpha_pred);
                                    maxupd = std::max(maxupd, this_upd);
                                }
                                if (maxupd > 0.0) std::cout << "[info] Applied predictor smoother (alpha=" << alpha_pred << ") max_update=" << maxupd << "\n";
                            }

                            const bool retry_ok = solve_current_kappa_stage();
                            if (!retry_ok) {
                                std::cout << "[error] Both omega and charge continuation failed for the same model (omega=" << omega << ", last_charge=" << ccharge << "). Exiting.\n";
                                return false;
                            }

                            // Omega retry succeeded: commit and resume omega continuation
                            omega = last_failed_omega_cand;
                            double current_charge = compute_current_charge();
                            charge_history.push_back(current_charge);
                            std::cout << "[info] Omega retry succeeded at omega=" << omega << "; resuming omega continuation.\n";
                            // write model and bs.dat as normal
                            if (!append_family_model("families.dat", false, omega)) std::cout << "[warn] Failed to append model to families.dat after omega retry.\n";
                            if (write_bs_dat()) std::cout << "[info] Wrote boson-star profile to bs.dat\n";

                            // break out of the persistent charge-continuation and resume
                            break;
                        }

                        // If charge advanced successfully, continue attempting further
                        // charge steps (we remain in charge mode until exhausted).
                        // Loop back and try another incremental charge step.
                        continue;
                    }

                    // Having handled charge continuation (either resumed omega or exited),
                    // continue outer flow: if we resumed omega, the outer omega loop
                    // will pick up from the updated 'omega' value; otherwise we returned.
                    break;
                } else {
                    break;
                }
            }
        }
    }

    return true;
}

bool BosonStarSolver::run_arclength_continuation() {
    // Pseudo-arclength continuation in the full discretized state x (all fields,
    // omega_s included). With x_prev, x_cur the last two solutions:
    //   t      = (x_cur - x_prev) / |x_cur - x_prev|      (secant tangent)
    //   x_pred = x_cur + ds * t                           (extrapolated seed)
    //   solve  R(x) = 0  together with  <t, x - x_pred> = 0,
    // the latter replacing the omega_s boundary condition at infinity. Every new
    // solution therefore lies a distance ds *ahead* of x_cur along t, so turning
    // points in omega or Q need no mode switch and the branch cannot be retraced.
    const size_t iAsym = static_cast<size_t>(p_.Nr - 2);
    auto vsub = [](const Vec& u, const Vec& w) {
        Vec d(u.size());
        for (size_t k = 0; k < u.size(); ++k) d[k] = u[k] - w[k];
        return d;
    };
    auto record = [&](const char* tag) {
        const double w = state_.omega_s[iAsym];
        std::cout << "[info] " << tag << ": omega = " << w << "  M = " << compute_current_mass()
                  << "  Q = " << compute_current_charge() << "  phi(0) = " << state_.phi[0] << "\n";
        if (!append_family_model("families.dat", false, w)) {
            std::cout << "[warn] Failed to append model to families.dat\n";
        }
        write_bs_dat();
    };

    // First point: the model already solved at omega_target (omega fixed).
    arclength_active_ = true;
    Vec x_prev;
    pack_state_vector(x_prev);
    const SolverState first_state = state_;

    // Second point: one ordinary omega step, to define the first secant.
    arclength_active_ = false;
    const double w0 = p_.omega_target;
    double dw = (p_.delta_omega > 0.0) ? p_.delta_omega : 1e-3;
    bool ok = false;
    for (int tries = 0; tries < 8 && !ok; ++tries) {
        state_ = first_state;
        p_.omega_target = w0 - dw;
        ok = solve_current_kappa_stage();
        if (!ok) dw *= 0.5;
    }
    if (!ok) {
        std::cout << "[warn] Arclength: could not take the initial omega step from omega = " << w0 << ".\n";
        return false;
    }
    arclength_active_ = true;
    Vec x_cur;
    pack_state_vector(x_cur);
    record("Arclength start");

    double ds = l2_norm(vsub(x_cur, x_prev));
    const double ds_max = (p_.arclength_ds_max > 0.0) ? p_.arclength_ds_max : 10.0 * ds;
    const double ds_min = 1e-6 * ds;
    ds = std::min(ds, ds_max);

    int accepted = 0;
    while (accepted < p_.arclength_max_steps) {
        const Vec sec = vsub(x_cur, x_prev);
        const double sec_norm = l2_norm(sec);
        arc_tangent_.resize(sec.size());
        for (size_t k = 0; k < sec.size(); ++k) arc_tangent_[k] = sec[k] / sec_norm;
        arc_pred_.resize(sec.size());
        for (size_t k = 0; k < sec.size(); ++k) arc_pred_[k] = x_cur[k] + ds * arc_tangent_[k];

        unpack_state_vector(arc_pred_);
        bool step_ok = solve_current_kappa_stage();
        const int iters = (last_iters_to_floor_ > 0) ? last_iters_to_floor_ : last_iterations_;

        Vec x_new;
        pack_state_vector(x_new);
        if (step_ok) {
            // Reject jumps to a distant part of the curve (e.g. across a tight
            // spiral): consecutive secants must stay roughly aligned.
            const Vec d = vsub(x_new, x_cur);
            const double dn = l2_norm(d);
            double cosang = 0.0;
            for (size_t k = 0; k < d.size(); ++k) cosang += d[k] * arc_tangent_[k];
            cosang /= std::max(1e-300, dn);
            if (cosang < p_.arclength_min_cos) {
                std::cout << "[warn] Arclength: secant turned too sharply (cos = " << cosang << "); reducing ds.\n";
                step_ok = false;
            }
        }
        if (!step_ok) {
            assign_state_raw(x_cur);
            ds *= 0.5;
            std::cout << "[info] Arclength: step failed, ds -> " << ds << "\n";
            if (ds < ds_min) {
                std::cout << "[warn] Arclength: step size underflow; stopping.\n";
                return false;
            }
            continue;
        }

        // Under-resolved profiles make the discrete branch fold back spuriously
        // (the fold moves with Nr), which continuation would then faithfully follow.
        // Two cases seen: deep in the spiral the isotropic-radius core shrinks onto
        // r = 0 (f jumps across the first cell); thin-wall stars have a wall only a
        // cell or two wide (phi jumps between neighbouring points).
        const double core_jump = state_.metric_f[1] / std::max(1e-300, state_.metric_f[0]) - 1.0;
        double phi_max = 0.0, phi_jump = 0.0;
        for (size_t k = 0; k < state_.phi.size(); ++k) {
            phi_max = std::max(phi_max, std::abs(state_.phi[k]));
            if (k + 1 < state_.phi.size()) phi_jump = std::max(phi_jump, std::abs(state_.phi[k + 1] - state_.phi[k]));
        }
        phi_jump /= std::max(1e-300, phi_max);
        if (core_jump > p_.core_resolution_tol || phi_jump > p_.wall_resolution_tol) {
            std::cout << "[info] Arclength: profile under-resolved (f(r1)/f(0) - 1 = " << core_jump
                      << ", max cell jump in phi / max phi = " << phi_jump
                      << "; tolerances " << p_.core_resolution_tol << ", " << p_.wall_resolution_tol
                      << "); model discarded, stopping. Increase Nr to continue.\n";
            assign_state_raw(x_cur);
            return true;
        }

        x_prev = x_cur;
        x_cur = x_new;
        ++accepted;
        record("Arclength step");

        const double w = state_.omega_s[iAsym];
        const double w_max = (p_.arclength_omega_max > 0.0) ? std::min(boson_mass(p_), p_.arclength_omega_max)
                                                            : boson_mass(p_);
        if (w <= p_.arclength_omega_min || w >= w_max) {
            std::cout << "[info] Arclength: omega = " << w << " left (" << p_.arclength_omega_min << ", " << w_max
                      << "); stopping.\n";
            return true;
        }

        // Step-size control from Newton effort.
        if (iters <= 3) ds = std::min(ds_max, 1.5 * ds);
        else if (iters >= 7) ds *= 0.7;
    }
    std::cout << "[info] Arclength: reached arclength_max_steps.\n";
    return true;
}

bool BosonStarSolver::solve_current_kappa_stage() {

    last_residual_norm_ = std::numeric_limits<double>::infinity();
    last_iterations_ = 0;
    last_iters_to_floor_ = 0;
    for (int it = 0; it < p_.max_newton_iters; ++it) {
        // Ensure any continuation-controlled parameter slices (e.g. omega)
        // in the in-memory state reflect the current targets before forming
        // the residual/Jacobian. In the omega-family continuation we update
        // p_.omega_target prior to calling this function; make sure the
        // state_.omega_s array is consistent so residuals and Jacobian
        // evaluations don't compare against mismatched frequency values.
        enforce_boundary_values();

        Vec x;
        pack_state_vector(x);

        const Residuals R = compute_residuals();
        Vec r;
        pack_residual_vector(R, r);
        const double res_norm = inf_norm(r);

    // (removed) per-block residual debug prints to reduce runtime verbosity.

        last_residual_norm_ = res_norm;
        last_iterations_ = it + 1;

    // suppressed per-iteration residual_inf prints to reduce log verbosity

        // Componentwise test |R_i| <= rel * scale_i + abs_tol_residual, where
        // scale_i is the sum of |terms| in row i. Round-off in R_i is ~eps*scale_i,
        // so this stays meaningful at high Nr (terms ~ 1/h^2) and for compact stars
        // (terms ~ 1/f^2 with f(0) << 1), where a fixed absolute tolerance is not.
        Vec sc;
        pack_scale_vector(R, sc);
        const bool at_floor = residual_within(r, sc, p_.stall_rel_tol);
        // Newton effort used for continuation step control: iterations needed to
        // reach the stall tolerance (later iterations only polish round-off).
        if (at_floor && last_iters_to_floor_ == 0) last_iters_to_floor_ = it + 1;
        if (residual_within(r, sc, p_.rel_tol_residual)) {
            return accept_converged_state("residual tolerance");
        }

        Vec rhs(r.size(), 0.0);
        for (size_t i = 0; i < r.size(); ++i) {
            rhs[i] = -r[i];
        }

        Vec delta(r.size(), 0.0);
        bool linear_ok = false;

        // Choose linear solver based on parameter
        if (p_.linear_solver == "banded") {
            // Colored finite-difference Jacobian + banded LU (see assemble_and_solve_banded).
            linear_ok = assemble_and_solve_banded(x, rhs, delta);
            if (!linear_ok) {
                std::cout << "[warn] Banded LU solve failed; falling back to GMRES.\n";
                linear_ok = gmres_solve(x, r, rhs, delta);
            }
        } else if (p_.linear_solver == "direct") {
            // Assemble Jacobian and solve via dense LU
            std::vector<double> J;
            if (!assemble_jacobian(x, J)) {
                std::cout << "[warn] Jacobian assembly failed; falling back to GMRES.\n";
                linear_ok = gmres_solve(x, r, rhs, delta);
            } else {
                const int n = static_cast<int>(rhs.size());
                std::vector<double> sol;
                // Solve J * delta = rhs
                if (!dense_lu_solve(J, rhs, sol, n)) {
                    std::cout << "[warn] LU solve failed; attempting small regularization and fallback to GMRES.\n";
                    // Try Tikhonov regularization: J += eps*I
                    const double reg = 1e-12;
                    for (int i = 0; i < n; ++i) J[static_cast<size_t>(i * n + i)] += reg;
                    if (!dense_lu_solve(J, rhs, sol, n)) {
                        std::cout << "[warn] Regularized LU solve also failed; using GMRES.\n";
                        linear_ok = gmres_solve(x, r, rhs, delta);
                    } else {
                        delta = sol;
                        linear_ok = true;
                    }
                } else {
                    delta = sol;
                    linear_ok = true;
                }
            }
        } else {
            linear_ok = gmres_solve(x, r, rhs, delta);
        }
        if (!linear_ok) {
            if (l2_norm(delta) < 1e-16) {
                // Last-resort fallback direction.
                delta = rhs;
            }
            std::cout << "[warn] GMRES did not reach linear tolerance; using current Krylov iterate.\n";
        }

        const double upd_norm = inf_norm(delta);
        const bool accepted = apply_newton_step_with_line_search(x, delta, res_norm);
        if (!accepted) {
            // At the round-off floor no step can reduce the residual; accept the
            // state if we are there, but only after Newton has taken at least one
            // step (an untouched continuation predictor is not a solution).
            if (it > 0 && at_floor) {
                std::cout << "[info] Residual " << res_norm << " stagnated within stall_rel_tol of its round-off scale.\n";
                return accept_converged_state("round-off floor");
            }
            std::cout << "[warn] Line-search failed; applying small residual-smoother fallback step.\n";
            const double fallback_upd = apply_residual_smoother_step(R, 1e-3);
            if (fallback_upd < p_.abs_tol_update) {
                return false;
            }
            continue;
        }

        if (upd_norm < p_.abs_tol_update) {
            // A vanishing update with a large residual is a stall, not convergence.
            if (!at_floor) {
                std::cout << "[warn] Newton update vanished with residual " << res_norm << "; not converged.\n";
                return false;
            }
            return accept_converged_state("update tolerance");
        }
    }

    return false;
}

void BosonStarSolver::pack_scale_vector(const Residuals& R, Vec& s) const {
    Residuals S;
    S.f = R.sf; S.l = R.sl; S.phi = R.sphi; S.omega = R.somega; S.rho = R.srho;
    pack_residual_vector(S, s);
}

bool BosonStarSolver::residual_within(const Vec& r, const Vec& scale, double rel) const {
    for (size_t i = 0; i < r.size(); ++i) {
        if (std::abs(r[i]) > rel * scale[i] + p_.abs_tol_residual) return false;
    }
    return true;
}

bool BosonStarSolver::accept_converged_state(const char* criterion) const {
    // phi = 0 with flat metric solves every equation for any omega, so a
    // "converged" state with vanishing scalar field is a failed solve.
    double max_phi = 0.0;
    for (double v : state_.phi) max_phi = std::max(max_phi, std::abs(v));
    if (max_phi < p_.trivial_phi_tol) {
        std::cout << "[warn] Converged (" << criterion << ") to the trivial vacuum, max|phi| = "
                  << max_phi << "; treating as failure.\n";
        return false;
    }
    // unpack_state_vector clamps f, l >= 1e-8. A state sitting on that floor is
    // a clamp artifact, typically a central lapse too small for the grid.
    const double fmin = *std::min_element(state_.metric_f.begin(), state_.metric_f.end());
    const double lmin = *std::min_element(state_.metric_l.begin(), state_.metric_l.end());
    if (std::min(fmin, lmin) < 1e-6) {
        std::cout << "[warn] Converged (" << criterion << ") with min(f, l) = " << std::min(fmin, lmin)
                  << " at the clamp floor (central lapse unresolved; increase Nr); treating as failure.\n";
        return false;
    }
    std::cout << "[info] Converged by " << criterion << ".\n";
    return true;
}

void BosonStarSolver::print_run_summary() const {
    std::cout << "\n==== Solver run summary ====\n";
    std::cout << "grid points (Nr) : " << state_.rbar.size() << "\n";
    std::cout << "iterations       : " << last_iterations_ << "\n";
    std::cout << "residual_inf     : " << last_residual_norm_ << "\n";
    if (!state_.phi.empty()) {
        std::cout << "phi(0) seed      : " << state_.phi.front() << "\n";
        std::cout << "phi(rmax) seed   : " << state_.phi.back() << "\n";
    }
    std::cout << "==============================\n";
}

void BosonStarSolver::build_grid() {
    state_.rbar.resize(static_cast<size_t>(p_.Nr));
    state_.r.resize(static_cast<size_t>(p_.Nr));

    const double dr = (p_.rbar_max - p_.rbar_min) / static_cast<double>(p_.Nr - 1);
    for (int i = 0; i < p_.Nr; ++i) {
        const size_t k = static_cast<size_t>(i);
        state_.rbar[k] = p_.rbar_min + dr * static_cast<double>(i);

        const double rb = state_.rbar[k];
        if (i == p_.Nr - 1) {
            state_.r[k] = 1.0e12; // practical proxy for infinity endpoint
        } else {
            const double denom = std::max(1e-12, 1.0 - rb);
            state_.r[k] = rb / denom;
        }
    }
}

void BosonStarSolver::apply_regular_seed_profile() {
    const size_t N = state_.rbar.size();

    state_.phi.assign(N, 0.0);
    state_.metric_f.assign(N, 1.0);
    state_.metric_l.assign(N, 1.0);
    state_.omega_s.assign(N, p_.omega_target);
    state_.rho.assign(N, 1.0);

    // Smooth spherical seed in compactified coordinate, shaped like a gravitating
    // mini boson star (used directly at kappa_target, see run_relaxation_skeleton).
    // Near the Newtonian limit phi(0) ~ eps^2 and the size ~ 1/eps with
    // eps^2 = 1 - omega^2/mu^2; the field equations depend on kappa*phi^2, hence
    // the sqrt(8 pi / kappa) factor. Amplitude, width and central f are fits to
    // mini boson stars at omega = 0.85-0.95 (1 - f(0) ~ 1.94 eps^2 there).
    const double mu = std::max(1e-12, boson_mass(p_));
    const double w_mu = std::min(0.99, std::max(0.5, p_.omega_target / mu));
    const double eps2 = 1.0 - w_mu * w_mu;
    const double kap_seed = (p_.kappa_target > 0.0) ? p_.kappa_target : 8.0 * kPi;
    const double phi_c = (0.14 + 0.2 * eps2) * eps2 * std::sqrt(8.0 * kPi / kap_seed);
    const double width2 = std::max(1.5, 6.5 / eps2 - 16.0) / (mu * mu);
    const double f_dip = std::min(0.8, 1.94 * eps2);
    for (size_t i = 0; i < N; ++i) {
        const double r = state_.r[i];

        state_.phi[i] = phi_c * std::exp(-r * r / width2);
        state_.metric_f[i] = 1.0 - f_dip / std::sqrt(1.0 + r * r / width2);
        state_.metric_l[i] = 1.0;

        // Placeholder for rho; we'll compute a physics-informed seed below.
        state_.rho[i] = 1.0;
    }

    // Compute a physics-informed rho seed by solving the radial equation
    // rho'' + (2/r) rho' = phi^2, which is the flat-metric approximation of
    // the auxiliary rho equation. Use Neumann at r=0 and Dirichlet rho(inf)=1.
    if (N >= 3) {
        const int M = static_cast<int>(N) - 1; // unknowns rho[0..M-1], rho[M]=rho[N-1]=1
        std::vector<double> a(M, 0.0), b(M, 0.0), c(M, 0.0), rhs(M, 0.0);

        // i=0 Neumann: rho1 - rho0 = 0
        b[0] = -1.0;
        c[0] = 1.0;
        rhs[0] = 0.0;

        for (int ii = 1; ii < M; ++ii) {
            const double r_im1 = state_.r[static_cast<size_t>(ii - 1)];
            const double r_i = state_.r[static_cast<size_t>(ii)];
            const double r_ip1 = state_.r[static_cast<size_t>(ii + 1)];

            const double h1 = r_i - r_im1;
            const double h2 = r_ip1 - r_i;
            const double denom = (h1 + h2);

            const double coef_im1 = 2.0 / denom * (1.0 / h1) - 2.0 / (r_i * denom);
            const double coef_i = 2.0 / denom * ( - (1.0 / h2 + 1.0 / h1) );
            const double coef_ip1 = 2.0 / denom * (1.0 / h2) + 2.0 / (r_i * denom);

            a[ii] = coef_im1;
            b[ii] = coef_i;
            c[ii] = coef_ip1;
            rhs[ii] = state_.phi[static_cast<size_t>(ii)] * state_.phi[static_cast<size_t>(ii)];
        }

        // account for known rho at last gridpoint (rho[N-1] = 1)
        if (M - 1 >= 1) {
            rhs[M - 1] -= c[M - 1] * 1.0;
            c[M - 1] = 0.0;
        }

        // Thomas algorithm (tridiagonal)
        std::vector<double> cp(M, 0.0), dp(M, 0.0);
        // forward
        const double safe_eps = 1e-14;
        cp[0] = c[0] / b[0];
        dp[0] = rhs[0] / b[0];
        for (int i = 1; i < M; ++i) {
            double denom_t = b[i] - a[i] * cp[i - 1];
            if (std::abs(denom_t) < safe_eps) denom_t = (denom_t >= 0 ? safe_eps : -safe_eps);
            cp[i] = c[i] / denom_t;
            dp[i] = (rhs[i] - a[i] * dp[i - 1]) / denom_t;
        }
        // backward
        std::vector<double> sol(M, 0.0);
        sol[M - 1] = dp[M - 1];
        for (int i = M - 2; i >= 0; --i) sol[i] = dp[i] - cp[i] * sol[i + 1];

        for (int i = 0; i < M; ++i) state_.rho[static_cast<size_t>(i)] = sol[i];
        state_.rho[static_cast<size_t>(N - 1)] = 1.0;
    }

    // Enforce outer/infinity boundary enforcement (origin handled above).
    enforce_boundary_values();
}

double BosonStarSolver::potential_U(double phi) const {
    return ::potential_U(p_, phi);
}

double BosonStarSolver::dU_dphi(double phi) const {
    return potential_dU(p_, phi);
}

double BosonStarSolver::d1_rbar(const std::vector<double>& u, int i, double h) const {
    const int N = static_cast<int>(u.size());
    // Use even symmetry at the origin: ghost point u[-1] = u[1]. This gives
    // d/d(rbar) u|_{0} = 0 for scalar fields with even symmetry.
    if (i <= 0) {
        return 0.0;
    }
    // Use a 4th-order five-point centered stencil where available (i in [2, N-3]).
    if (i >= 2 && i <= N - 3) {
        return fivePointDeriv(h, 1,
                              u[i - 2], u[i - 1], u[i], u[i + 1], u[i + 2]);
    }
    if (i >= N - 1) {
        // backward difference at outer boundary
        return (u[N - 1] - u[N - 2]) / h;
    }
    // fallback to 2nd-order centered at near-boundary points (i==1 or i==N-2)
    return (u[i + 1] - u[i - 1]) / (2.0 * h);
}

double BosonStarSolver::d2_rbar(const std::vector<double>& u, int i, double h) const {
    const int N = static_cast<int>(u.size());
    if (i <= 0) {
        // use ghost symmetry u[-1] = u[1]
        // second derivative at origin: (u[1] - 2 u[0] + u[-1]) / h^2 = 2*(u[1]-u[0])/h^2
        return 2.0 * (u[1] - u[0]) / (h * h);
    }
    // Use a 4th-order five-point second-derivative stencil where available
    if (i >= 2 && i <= N - 3) {
        return fivePointDeriv(h, 2,
                              u[i - 2], u[i - 1], u[i], u[i + 1], u[i + 2]);
    }
    if (i >= N - 1) {
        return 0.0;
    }
    // fallback to 2nd-order centered at near-boundary points
    return (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (h * h);
}

double BosonStarSolver::d1_rbar_abs(const std::vector<double>& u, int i, double h) const {
    // Same stencil as d1_rbar with |coefficients| * |u|: the scale of its rounding error.
    const int N = static_cast<int>(u.size());
    if (i <= 0) return 0.0;
    if (i >= 2 && i <= N - 3) {
        return ((1.0 / 12.0) * std::abs(u[i - 2]) + (2.0 / 3.0) * std::abs(u[i - 1])
                + (2.0 / 3.0) * std::abs(u[i + 1]) + (1.0 / 12.0) * std::abs(u[i + 2])) / h;
    }
    if (i >= N - 1) return (std::abs(u[N - 1]) + std::abs(u[N - 2])) / h;
    return (std::abs(u[i + 1]) + std::abs(u[i - 1])) / (2.0 * h);
}

double BosonStarSolver::d2_rbar_abs(const std::vector<double>& u, int i, double h) const {
    const int N = static_cast<int>(u.size());
    if (i <= 0) return 2.0 * (std::abs(u[1]) + std::abs(u[0])) / (h * h);
    if (i >= 2 && i <= N - 3) {
        return ((1.0 / 12.0) * std::abs(u[i - 2]) + (4.0 / 3.0) * std::abs(u[i - 1]) + 2.5 * std::abs(u[i])
                + (4.0 / 3.0) * std::abs(u[i + 1]) + (1.0 / 12.0) * std::abs(u[i + 2])) / (h * h);
    }
    if (i >= N - 1) return 0.0;
    return (std::abs(u[i + 1]) + 2.0 * std::abs(u[i]) + std::abs(u[i - 1])) / (h * h);
}

double BosonStarSolver::dU_dphi_abs_terms(double phi) const {
    return potential_dU_abs_terms(p_, phi);
}

double BosonStarSolver::d1_r_from_rbar(double u_b, double rb) const {
    const double one_m_rb = std::max(1e-12, 1.0 - rb);
    return one_m_rb * one_m_rb * u_b;
}

double BosonStarSolver::d2_r_from_rbar(double u_b, double u_bb, double rb) const {
    const double one_m_rb = std::max(1e-12, 1.0 - rb);
    const double one_m_rb2 = one_m_rb * one_m_rb;
    const double one_m_rb3 = one_m_rb2 * one_m_rb;
    const double one_m_rb4 = one_m_rb2 * one_m_rb2;
    return one_m_rb4 * u_bb - 2.0 * one_m_rb3 * u_b;
}

BosonStarSolver::Residuals BosonStarSolver::compute_residuals() const {
    const int N = p_.Nr;
    const double h = (p_.rbar_max - p_.rbar_min) / static_cast<double>(N - 1);

    Residuals R;
    R.f.assign(static_cast<size_t>(N), 0.0);
    R.l.assign(static_cast<size_t>(N), 0.0);
    R.phi.assign(static_cast<size_t>(N), 0.0);
    R.omega.assign(static_cast<size_t>(N), 0.0);
    R.rho.assign(static_cast<size_t>(N), 0.0);
    // Per-row scale: sum of |terms| entering each residual (rounding-error scale).
    R.sf.assign(static_cast<size_t>(N), 0.0);
    R.sl.assign(static_cast<size_t>(N), 0.0);
    R.sphi.assign(static_cast<size_t>(N), 0.0);
    R.somega.assign(static_cast<size_t>(N), 0.0);
    R.srho.assign(static_cast<size_t>(N), 0.0);

    // Interior equations: Appendix B.1 equations (59)-(63), written as residual = LHS-RHS.
    for (int i = 1; i < N - 1; ++i) {
        const double rb = state_.rbar[static_cast<size_t>(i)];
        const double r = std::max(1e-12, state_.r[static_cast<size_t>(i)]);

        const double f = std::max(1e-10, state_.metric_f[static_cast<size_t>(i)]);
        const double l = std::max(1e-10, state_.metric_l[static_cast<size_t>(i)]);
        const double phi = state_.phi[static_cast<size_t>(i)];
        const double omega = state_.omega_s[static_cast<size_t>(i)];
    const double kap = current_kappa_;

        const double f_b = d1_rbar(state_.metric_f, i, h);
        const double f_bb = d2_rbar(state_.metric_f, i, h);
        const double l_b = d1_rbar(state_.metric_l, i, h);
        const double l_bb = d2_rbar(state_.metric_l, i, h);
        const double p_b = d1_rbar(state_.phi, i, h);
        const double p_bb = d2_rbar(state_.phi, i, h);
        const double rho_b = d1_rbar(state_.rho, i, h);
        const double rho_bb = d2_rbar(state_.rho, i, h);
        const double w_b = d1_rbar(state_.omega_s, i, h);
        const double w_bb = d2_rbar(state_.omega_s, i, h);

        const double f_r = d1_r_from_rbar(f_b, rb);
        const double f_rr = d2_r_from_rbar(f_b, f_bb, rb);
        const double l_r = d1_r_from_rbar(l_b, rb);
        const double l_rr = d2_r_from_rbar(l_b, l_bb, rb);
        const double p_r = d1_r_from_rbar(p_b, rb);
        const double p_rr = d2_r_from_rbar(p_b, p_bb, rb);
        const double rho_r = d1_r_from_rbar(rho_b, rb);
        const double rho_rr = d2_r_from_rbar(rho_b, rho_bb, rb);
        const double w_r = d1_r_from_rbar(w_b, rb);
        const double w_rr = d2_r_from_rbar(w_b, w_bb, rb);

        const double U = potential_U(phi);
        const double dU = dU_dphi(phi);

        // Eq. (59): f'' term
        const double rhs_f =
            -0.5 * (1.0 / (f * l)) *
            (4.0 * kap * f * l * l * U - 8.0 * kap * omega * omega * l * l * phi * phi
             + (4.0 / r) * f * l * f_r - 2.0 * l * f_r * f_r + f * l_r * f_r);

        // Eq. (60): l'' term
        const double rhs_l =
            -0.5 * (1.0 / (f * l)) *
            (8.0 * kap * l * l * l * U - 8.0 * kap * omega * omega * (l * l * l / f) * phi * phi
             + (6.0 / r) * f * l * l_r - f * l_r * l_r);

        // Eq. (61): scalar field (match Appendix B.1 of Kleihaus et al.)
        // Note signs: rhs contains +2*omega^2*(l^2/f)*phi - l^2*dU + ... inside
        // the overall -0.5/(f*l) factor.
        const double rhs_phi =
            -0.5 * (1.0 / (f * l)) *
            ((2.0 * omega * omega * l * l / f) * phi - l * l * dU
             + (4.0 / r) * f * l * p_r + f * l_r * p_r);

        // Eq. (62): rho auxiliary
        const double eq_rho =
            -(f / l) * (rho_rr + (2.0 / r) * rho_r + 0.5 * (l_r / l) * rho_r)
            + (1.0 / f) * phi * phi;

        // Eq. (63): omega_s auxiliary (for charge-continuation mode)
        const double eq_omega =
            (f / l) * (w_rr + (2.0 / r) * w_r + 0.5 * (l_r / l) * w_r);

        R.f[static_cast<size_t>(i)] = f_rr - rhs_f;
        R.l[static_cast<size_t>(i)] = l_rr - rhs_l;
        R.phi[static_cast<size_t>(i)] = p_rr - rhs_phi;
        R.rho[static_cast<size_t>(i)] = eq_rho;
        R.omega[static_cast<size_t>(i)] = is_omega_continuation() ? 0.0 : eq_omega;

        // Row scales: same expressions with every term in absolute value and
        // derivatives replaced by their absolute stencils.
        const double om = std::max(1e-12, 1.0 - rb);
        const double om2 = om * om, om3 = om2 * om, om4 = om2 * om2;
        auto a1 = [&](const std::vector<double>& u) { return om2 * d1_rbar_abs(u, i, h); };
        auto a2 = [&](const std::vector<double>& u) { return om4 * d2_rbar_abs(u, i, h) + 2.0 * om3 * d1_rbar_abs(u, i, h); };
        const double fa1 = a1(state_.metric_f), fa2 = a2(state_.metric_f);
        const double la1 = a1(state_.metric_l), la2 = a2(state_.metric_l);
        const double pa1 = a1(state_.phi), pa2 = a2(state_.phi);
        const double ra1 = a1(state_.rho), ra2 = a2(state_.rho);
        const double wa1 = a1(state_.omega_s), wa2 = a2(state_.omega_s);
        const double pref = 0.5 / (f * l);
        const double w2 = omega * omega, p2 = phi * phi;
        R.sf[static_cast<size_t>(i)] = fa2 + pref *
            (std::abs(4.0 * kap * f * l * l * U) + 8.0 * kap * w2 * l * l * p2 + (4.0 / r) * f * l * fa1
             + 4.0 * l * std::abs(f_r) * fa1 + f * (la1 * std::abs(f_r) + std::abs(l_r) * fa1));
        R.sl[static_cast<size_t>(i)] = la2 + pref *
            (std::abs(8.0 * kap * l * l * l * U) + 8.0 * kap * w2 * (l * l * l / f) * p2
             + (6.0 / r) * f * l * la1 + 2.0 * f * std::abs(l_r) * la1);
        R.sphi[static_cast<size_t>(i)] = pa2 + pref *
            ((2.0 * w2 * l * l / f) * std::abs(phi) + l * l * dU_dphi_abs_terms(phi) + (4.0 / r) * f * l * pa1
             + f * (la1 * std::abs(p_r) + std::abs(l_r) * pa1));
        R.srho[static_cast<size_t>(i)] = (f / l) * (ra2 + (2.0 / r) * ra1 + 0.5 * (la1 * std::abs(rho_r) + std::abs(l_r) * ra1) / l) + p2 / f;
        R.somega[static_cast<size_t>(i)] = is_omega_continuation() ? 1.0
            : (f / l) * (wa2 + (2.0 / r) * wa1 + 0.5 * (la1 * std::abs(w_r) + std::abs(l_r) * wa1) / l);
    }

    // r = 0 boundary conditions: impose derivative-based Neumann BCs
    // using a one-sided second-order accurate forward difference in rbar
    // to ensure the origin value participates in the Newton solve. This
    // enforces du/drbar|_{0} = 0 -> (-3 u0 + 4 u1 - u2)/(2 h) = 0.
    if (N >= 3) {
        const double inv2h = 1.0 / (2.0 * h);
        if (is_lapse_continuation()) {
            // Enforce central metric_f value via lapse_target (stored as sqrt(f0)).
            // We keep lapse_target as sqrt(metric_f[0]) for user-facing semantics,
            // but enforce the squared value against metric_f directly so the
            // residual is linear in metric_f.
            const double f0_target = p_.lapse_target * p_.lapse_target;
            R.f[0] = state_.metric_f[0] - f0_target;
        } else {
            R.f[0] = (-3.0 * state_.metric_f[0] + 4.0 * state_.metric_f[1] - state_.metric_f[2]) * inv2h;
        }
        R.l[0] = (-3.0 * state_.metric_l[0] + 4.0 * state_.metric_l[1] - state_.metric_l[2]) * inv2h;
        R.phi[0] = (-3.0 * state_.phi[0] + 4.0 * state_.phi[1] - state_.phi[2]) * inv2h;
        R.rho[0] = (-3.0 * state_.rho[0] + 4.0 * state_.rho[1] - state_.rho[2]) * inv2h;
        R.omega[0] = (-3.0 * state_.omega_s[0] + 4.0 * state_.omega_s[1] - state_.omega_s[2]) * inv2h;
        auto s0 = [&](const std::vector<double>& u) {
            return (3.0 * std::abs(u[0]) + 4.0 * std::abs(u[1]) + std::abs(u[2])) * inv2h;
        };
        R.sf[0] = is_lapse_continuation() ? std::abs(state_.metric_f[0]) + p_.lapse_target * p_.lapse_target
                                          : s0(state_.metric_f);
        R.sl[0] = s0(state_.metric_l);
        R.sphi[0] = s0(state_.phi);
        R.srho[0] = s0(state_.rho);
        R.somega[0] = s0(state_.omega_s);
    } else {
        // fallback to existing rbar-derivative helper (which returns 0 for i<=0)
        R.f[0] = d1_rbar(state_.metric_f, 0, h);
        R.l[0] = d1_rbar(state_.metric_l, 0, h);
        R.phi[0] = d1_rbar(state_.phi, 0, h);
        R.rho[0] = d1_rbar(state_.rho, 0, h);
        R.omega[0] = d1_rbar(state_.omega_s, 0, h);
    }

    // Infinity boundary conditions: Eq. (30) + Appendix A frequency/charge mode.
    const int iInf = N - 1;
    const int iAsym = N - 2;
    const double rAsym = std::max(1e-6, state_.r[static_cast<size_t>(iAsym)]);

    R.f[static_cast<size_t>(iInf)] = state_.metric_f[static_cast<size_t>(iInf)] - 1.0;
    R.l[static_cast<size_t>(iInf)] = state_.metric_l[static_cast<size_t>(iInf)] - 1.0;
    R.phi[static_cast<size_t>(iInf)] = state_.phi[static_cast<size_t>(iInf)] - 0.0;
    R.sf[static_cast<size_t>(iInf)] = std::abs(state_.metric_f[static_cast<size_t>(iInf)]) + 1.0;
    R.sl[static_cast<size_t>(iInf)] = std::abs(state_.metric_l[static_cast<size_t>(iInf)]) + 1.0;
    R.sphi[static_cast<size_t>(iInf)] = std::abs(state_.phi[static_cast<size_t>(iInf)]);

    // rho boundary: during lapse continuation, impose an independent
    // asymptotic Neumann condition (not a duplicate of interior eqn) to keep
    // the system full-rank while still avoiding fixed Dirichlet rho(inf).
    if (is_lapse_continuation()) {
        R.rho[static_cast<size_t>(iInf)] = (state_.rho[static_cast<size_t>(iInf)] - state_.rho[static_cast<size_t>(iAsym)]) / h;
        R.srho[static_cast<size_t>(iInf)] = (std::abs(state_.rho[static_cast<size_t>(iInf)]) + std::abs(state_.rho[static_cast<size_t>(iAsym)])) / h;
    } else {
        R.rho[static_cast<size_t>(iInf)] = state_.rho[static_cast<size_t>(iInf)] - 1.0;
        R.srho[static_cast<size_t>(iInf)] = std::abs(state_.rho[static_cast<size_t>(iInf)]) + 1.0;
    }

    // omega boundary: in omega continuation fix asymptotic omega to target;
    // in lapse continuation enforce the coupled ratio constraint
    // omega_asym / lapse_center = lapse_ratio_target_.
    if (is_omega_continuation()) {
        R.omega[static_cast<size_t>(iInf)] = state_.omega_s[static_cast<size_t>(iInf)] - p_.omega_target;
        R.somega[static_cast<size_t>(iInf)] = std::abs(state_.omega_s[static_cast<size_t>(iInf)]) + std::abs(p_.omega_target);
    } else if (is_lapse_continuation()) {
        const double lapse_c = std::sqrt(std::max(1e-16, state_.metric_f[0]));
        const double w_asym = state_.omega_s[static_cast<size_t>(iAsym)];
        R.omega[static_cast<size_t>(iInf)] = (w_asym / lapse_c) - lapse_ratio_target_;
        R.somega[static_cast<size_t>(iInf)] = std::abs(w_asym / lapse_c) + std::abs(lapse_ratio_target_);
    } else if (is_arclength_continuation()) {
        // omega_s is a free (constant) field; this row selects which point of
        // the solution curve we want instead of fixing omega or Q.
        R.omega[static_cast<size_t>(iInf)] = arclength_residual(&R.somega[static_cast<size_t>(iInf)]);
    } else {
        const double hAsym = h;
        const double rbAsym = state_.rbar[static_cast<size_t>(iAsym)];
        const double rho_b_asym = d1_rbar(state_.rho, iAsym, hAsym);
        const double rho_r_asym = d1_r_from_rbar(rho_b_asym, rbAsym);
        const double w_asym = state_.omega_s[static_cast<size_t>(iAsym)];
        R.omega[static_cast<size_t>(iInf)] = 8.0 * kPi * rAsym * rAsym * w_asym * rho_r_asym - p_.charge_target;
        const double rho_r_abs = d1_r_from_rbar(d1_rbar_abs(state_.rho, iAsym, hAsym), rbAsym);
        R.somega[static_cast<size_t>(iInf)] = 8.0 * kPi * rAsym * rAsym * std::abs(w_asym) * rho_r_abs + std::abs(p_.charge_target);
    }

    // Debug: print rho-equation term-by-term once (to avoid spamming during
    // iterative linear solves). This helps identify whether the Laplacian or
    // the source term (1/f * phi^2) dominates the rho residual.
    static bool rho_debug_printed = false;
    if (!rho_debug_printed) {
        rho_debug_printed = true;
        auto print_rho_terms = [&](int idx) {
            const int j = std::max(0, std::min(static_cast<int>(N - 1), idx));
            const double rb_j = state_.rbar[static_cast<size_t>(j)];
            const double r_j = std::max(1e-12, state_.r[static_cast<size_t>(j)]);
            const double f_j = std::max(1e-10, state_.metric_f[static_cast<size_t>(j)]);
            const double l_j = std::max(1e-10, state_.metric_l[static_cast<size_t>(j)]);
            const double phi_j = state_.phi[static_cast<size_t>(j)];

            const double rho_b_j = d1_rbar(state_.rho, j, h);
            const double rho_bb_j = d2_rbar(state_.rho, j, h);
            const double rho_r_j = d1_r_from_rbar(rho_b_j, rb_j);
            const double rho_rr_j = d2_r_from_rbar(rho_b_j, rho_bb_j, rb_j);

            const double l_b_j = d1_rbar(state_.metric_l, j, h);
            const double lap_j = -(f_j / l_j) * (rho_rr_j + (2.0 / r_j) * rho_r_j + 0.5 * (l_b_j / l_j) * rho_r_j);
            const double src_j = (1.0 / f_j) * phi_j * phi_j;
            const double eq_rho_j = lap_j + src_j;

            std::cout << "[debug-rho] idx=" << j
                      << " r=" << r_j
                      << " lap=" << lap_j
                      << " src=" << src_j
                      << " resid=" << eq_rho_j
                      << " (phi=" << phi_j << ")\n";
        };

        // choose indices: origin, midpoint, and phi-peak
        const int i_mid = N / 2;
        int i_peak = 0;
        double maxphi = 0.0;
        for (int ii = 0; ii < N; ++ii) {
            if (std::abs(state_.phi[static_cast<size_t>(ii)]) > maxphi) {
                maxphi = std::abs(state_.phi[static_cast<size_t>(ii)]);
                i_peak = ii;
            }
        }
        std::cout << "[debug-rho] per-term diagnostics (one-shot)\n";
        print_rho_terms(0);
        print_rho_terms(i_mid);
        print_rho_terms(i_peak);

        // Additional diagnostic: compare 2nd-order rbar-stencil -> physical derivatives
        // against 4th-order (five-point) stencil mapped to physical r derivatives.
        // Use a smooth analytic test function u(r)=exp(-r^2).
        double err_pr_2_rms = 0.0, err_pr_4_rms = 0.0;
        double err_prr_2_rms = 0.0, err_prr_4_rms = 0.0;
        double err_pr_2_max = 0.0, err_pr_4_max = 0.0;
        double err_prr_2_max = 0.0, err_prr_4_max = 0.0;
        int count = 0;
        std::vector<double> u(N);
        for (int ii = 0; ii < N; ++ii) u[ii] = std::exp(-state_.r[static_cast<size_t>(ii)] * state_.r[static_cast<size_t>(ii)]);
        for (int ii = 2; ii < N - 2; ++ii) {
            const double rb = state_.rbar[static_cast<size_t>(ii)];
            const double r = state_.r[static_cast<size_t>(ii)];
            const double h_local = h;

            // second-order rbar derivatives
            const double ub_b_2 = d1_rbar(u, ii, h_local);
            const double ub_bb_2 = d2_rbar(u, ii, h_local);
            const double pr_2 = d1_r_from_rbar(ub_b_2, rb);
            const double prr_2 = d2_r_from_rbar(ub_b_2, ub_bb_2, rb);

            // fourth-order rbar derivatives via fivePointDeriv
            const double ub_b_4 = fivePointDeriv(h_local, 1, u[ii - 2], u[ii - 1], u[ii], u[ii + 1], u[ii + 2]);
            const double ub_bb_4 = fivePointDeriv(h_local, 2, u[ii - 2], u[ii - 1], u[ii], u[ii + 1], u[ii + 2]);
            const double pr_4 = d1_r_from_rbar(ub_b_4, rb);
            const double prr_4 = d2_r_from_rbar(ub_b_4, ub_bb_4, rb);

            // analytic derivatives of u(r)=exp(-r^2)
            const double u_analytic = std::exp(-r * r);
            const double pr_exact = -2.0 * r * u_analytic;
            const double prr_exact = (4.0 * r * r - 2.0) * u_analytic;

            const double e_pr2 = std::abs(pr_2 - pr_exact);
            const double e_pr4 = std::abs(pr_4 - pr_exact);
            const double e_prr2 = std::abs(prr_2 - prr_exact);
            const double e_prr4 = std::abs(prr_4 - prr_exact);

            err_pr_2_rms += e_pr2 * e_pr2;
            err_pr_4_rms += e_pr4 * e_pr4;
            err_prr_2_rms += e_prr2 * e_prr2;
            err_prr_4_rms += e_prr4 * e_prr4;
            err_pr_2_max = std::max(err_pr_2_max, e_pr2);
            err_pr_4_max = std::max(err_pr_4_max, e_pr4);
            err_prr_2_max = std::max(err_prr_2_max, e_prr2);
            err_prr_4_max = std::max(err_prr_4_max, e_prr4);
            ++count;
        }
        if (count > 0) {
            err_pr_2_rms = std::sqrt(err_pr_2_rms / count);
            err_pr_4_rms = std::sqrt(err_pr_4_rms / count);
            err_prr_2_rms = std::sqrt(err_prr_2_rms / count);
            err_prr_4_rms = std::sqrt(err_prr_4_rms / count);
            std::cout << "[deriv-test] pr RMS errors: 2nd=" << err_pr_2_rms << " 4th=" << err_pr_4_rms << " max:2nd=" << err_pr_2_max << " 4th=" << err_pr_4_max << "\n";
            std::cout << "[deriv-test] prr RMS errors:2nd=" << err_prr_2_rms << " 4th=" << err_prr_4_rms << " max:2nd=" << err_prr_2_max << " 4th=" << err_prr_4_max << "\n";
        }
    }

    return R;
}

void BosonStarSolver::pack_residual_vector(const Residuals& R, Vec& r) const {
    r.assign(state_vector_size(), 0.0);
    const size_t N = static_cast<size_t>(p_.Nr);

    size_t off = 0;
    for (size_t i = 0; i < N; ++i) r[off + i] = R.f[i];
    off += N;
    for (size_t i = 0; i < N; ++i) r[off + i] = R.l[i];
    off += N;
    for (size_t i = 0; i < N; ++i) r[off + i] = R.phi[i];
    off += N;

    if (omega_is_unknown()) {
        for (size_t i = 0; i < N; ++i) r[off + i] = R.omega[i];
        off += N;
    }

    for (size_t i = 0; i < N; ++i) r[off + i] = R.rho[i];
}

void BosonStarSolver::pack_state_vector(Vec& x) const {
    x.assign(state_vector_size(), 0.0);
    const size_t N = static_cast<size_t>(p_.Nr);

    size_t off = 0;
    for (size_t i = 0; i < N; ++i) x[off + i] = state_.metric_f[i];
    off += N;
    for (size_t i = 0; i < N; ++i) x[off + i] = state_.metric_l[i];
    off += N;
    for (size_t i = 0; i < N; ++i) x[off + i] = state_.phi[i];
    off += N;

    if (omega_is_unknown()) {
        for (size_t i = 0; i < N; ++i) x[off + i] = state_.omega_s[i];
        off += N;
    }

    for (size_t i = 0; i < N; ++i) x[off + i] = state_.rho[i];
}

void BosonStarSolver::unpack_state_vector(const Vec& x) {
    const size_t N = static_cast<size_t>(p_.Nr);
    size_t off = 0;

    for (size_t i = 0; i < N; ++i) state_.metric_f[i] = std::max(1e-8, x[off + i]);
    off += N;
    for (size_t i = 0; i < N; ++i) state_.metric_l[i] = std::max(1e-8, x[off + i]);
    off += N;
    for (size_t i = 0; i < N; ++i) state_.phi[i] = x[off + i];
    off += N;

    if (omega_is_unknown()) {
        for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x[off + i];
        off += N;
    } else {
        std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
    }

    for (size_t i = 0; i < N; ++i) state_.rho[i] = x[off + i];
}

size_t BosonStarSolver::active_field_count() const {
    return static_cast<size_t>(omega_is_unknown() ? 5 : 4);
}

size_t BosonStarSolver::state_vector_size() const {
    return active_field_count() * static_cast<size_t>(p_.Nr);
}

double BosonStarSolver::max_abs_residual(const Residuals& R) const {
    double m = 0.0;
    auto upd = [&m](const std::vector<double>& v) {
        for (double x : v) {
            m = std::max(m, std::abs(x));
        }
    };
    upd(R.f);
    upd(R.l);
    upd(R.phi);
    upd(R.omega);
    upd(R.rho);
    return m;
}

double BosonStarSolver::inf_norm(const Vec& v) const {
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::abs(x));
    }
    return m;
}

double BosonStarSolver::l2_norm(const Vec& v) const {
    double s = 0.0;
    for (double x : v) {
        s += x * x;
    }
    return std::sqrt(s);
}

void BosonStarSolver::jacobian_vector_product_fd(
    const Vec& x_base,
    const Vec& r_base,
    const Vec& v,
    Vec& jv) {
    const double nv = l2_norm(v);
    if (nv < 1e-18) {
        jv.assign(v.size(), 0.0);
        return;
    }

    const double nx = std::max(1.0, l2_norm(x_base));
    double eps = p_.jacobian_fd_eps * nx / nv;
    // Clamp FD step to a safe window to avoid leaving the linear regime
    eps = std::max(1e-12, std::min(1e-3, eps));

    Vec x_pert = x_base;
    for (size_t i = 0; i < x_pert.size(); ++i) {
        x_pert[i] += eps * v[i];
    }

    // Apply perturbed state directly to avoid the clamping performed by
    // unpack_state_vector(), which introduces nondifferentiable kinks that
    // contaminate finite-difference Jacobian-vector products.
    {
        const size_t N = static_cast<size_t>(p_.Nr);
        size_t off = 0;
        for (size_t i = 0; i < N; ++i) state_.metric_f[i] = x_pert[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.metric_l[i] = x_pert[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.phi[i] = x_pert[off + i];
        off += N;
        if (omega_is_unknown()) {
            for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x_pert[off + i];
            off += N;
        } else {
            std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
        }
        for (size_t i = 0; i < N; ++i) state_.rho[i] = x_pert[off + i];
    }
    const Residuals Rp = compute_residuals();
    Vec rp;
    pack_residual_vector(Rp, rp);

    jv.assign(rp.size(), 0.0);
    for (size_t i = 0; i < rp.size(); ++i) {
        jv[i] = (rp[i] - r_base[i]) / eps;
    }

    // restore base state (direct assignment to avoid reintroducing clamps)
    {
        const size_t N = static_cast<size_t>(p_.Nr);
        size_t off = 0;
        for (size_t i = 0; i < N; ++i) state_.metric_f[i] = x_base[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.metric_l[i] = x_base[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.phi[i] = x_base[off + i];
        off += N;
        if (omega_is_unknown()) {
            for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x_base[off + i];
            off += N;
        } else {
            std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
        }
        for (size_t i = 0; i < N; ++i) state_.rho[i] = x_base[off + i];
    }
}

bool BosonStarSolver::gmres_solve(
    const Vec& x_base,
    const Vec& r_base,
    const Vec& rhs,
    Vec& delta) {
    const size_t n = rhs.size();
    const int m = std::max(2, p_.gmres_restart);
    const int max_it = std::max(m, p_.gmres_max_iters);

    delta.assign(n, 0.0);
    // Block-Jacobi preconditioner base: estimate per-row scaling by applying
    // the Jacobian to a block-wise ones vector for each active field block
    // to capture differences in physical scales. Additionally, build a
    // tridiagonal approximate inverse for the rho-block (dominant Laplacian)
    // and use that as the rho-subblock preconditioner.
    Vec Mdiag(n, 1.0);
    const size_t N = static_cast<size_t>(p_.Nr);
    const size_t n_blocks = active_field_count();
    size_t off_block = 0;
    for (size_t b = 0; b < n_blocks; ++b) {
        // build ones-in-block vector
        Vec vb(n, 0.0);
        for (size_t j = 0; j < N; ++j) {
            vb[off_block + j] = 1.0;
        }
        Vec Je;
        jacobian_vector_product_fd(x_base, r_base, vb, Je);
        for (size_t i = 0; i < n; ++i) {
            Mdiag[i] = std::max(Mdiag[i], 1.0 + std::abs(Je[i]));
        }
        off_block += N;
    }

    // Build rho-block tridiagonal preconditioner (size N). We'll form a
    // simple discretization of the radial operator used in the rho-equation
    // similar to the coefficients used when generating the physics-informed
    // rho seed. Last row uses mode-consistent asymptotic BC.
    std::vector<double> rho_a(N, 0.0), rho_b(N, 0.0), rho_c(N, 0.0);
    for (size_t ii = 0; ii < N; ++ii) {
        if (ii == 0) {
            // Neumann at origin: rho1 - rho0 = 0
            rho_a[ii] = 0.0;
            rho_b[ii] = -1.0;
            rho_c[ii] = 1.0;
            continue;
        }
        if (ii == N - 1) {
            if (is_lapse_continuation()) {
                // Lapse mode: asymptotic Neumann rho_N-1 - rho_N-2 = 0
                rho_a[ii] = -1.0;
                rho_b[ii] = 1.0;
                rho_c[ii] = 0.0;
            } else {
                // Dirichlet at infinity: rho_{N-1} = 1
                rho_a[ii] = 0.0;
                rho_b[ii] = 1.0;
                rho_c[ii] = 0.0;
            }
            continue;
        }

        const double r_im1 = state_.r[ii - 1];
        const double r_i = state_.r[ii];
        const double r_ip1 = state_.r[ii + 1];
        const double h1 = r_i - r_im1;
        const double h2 = r_ip1 - r_i;
        const double denom = (h1 + h2);

        // coefficients approximating rho'' + (2/r) rho' (from nonuniform FD)
        const double coef_im1 = 2.0 / denom * (1.0 / h1) - 2.0 / (r_i * denom);
        const double coef_i = 2.0 / denom * ( - (1.0 / h2 + 1.0 / h1) );
        const double coef_ip1 = 2.0 / denom * (1.0 / h2) + 2.0 / (r_i * denom);

        // First-derivative nonuniform stencil coefficients for rho_r
        const double a_fd = -h2 / (h1 * denom);
        const double b_fd = (h2 - h1) / (h1 * h2);
        const double c_fd = h1 / (h2 * denom);

        const double f_i = std::max(1e-12, state_.metric_f[ii]);
        const double l_i = std::max(1e-12, state_.metric_l[ii]);

        // Compute l_r at node ii using rbar derivatives -> r derivatives mapping
        const double h_rbar = (p_.rbar_max - p_.rbar_min) / static_cast<double>(p_.Nr - 1);
        const double l_b = d1_rbar(state_.metric_l, static_cast<int>(ii), h_rbar);
        const double l_r = d1_r_from_rbar(l_b, state_.rbar[static_cast<size_t>(ii)]);

        // Contribution from 0.5*(l_r/l) * rho_r: multiply first-derivative stencil by alpha
        const double alpha = 0.5 * (l_r / l_i);

        // approximate operator: -(f/l)*(rho'' + (2/r) rho') + 0.5*(l_r/l)*rho'
        rho_a[ii] = -(f_i / l_i) * coef_im1 + alpha * a_fd;
        rho_b[ii] = -(f_i / l_i) * coef_i + alpha * b_fd;
        rho_c[ii] = -(f_i / l_i) * coef_ip1 + alpha * c_fd;
    }

    // Helper: solve tridiagonal system (a: lower, b: diag, c: upper) for size N
    auto solve_tridiag = [&](const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, const Vec& rhs_r, Vec& out) {
        const double safe_eps = 1e-14;
        out.assign(static_cast<size_t>(N), 0.0);
        std::vector<double> cp(static_cast<size_t>(N), 0.0), dp(static_cast<size_t>(N), 0.0);
        double denom0 = b[0];
        if (std::abs(denom0) < safe_eps) denom0 = (denom0 >= 0 ? safe_eps : -safe_eps);
        cp[0] = c[0] / denom0;
        dp[0] = rhs_r[0] / denom0;
        for (size_t i = 1; i < static_cast<size_t>(N); ++i) {
            double denom = b[i] - a[i] * cp[i - 1];
            if (std::abs(denom) < safe_eps) denom = (denom >= 0 ? safe_eps : -safe_eps);
            cp[i] = c[i] / denom;
            dp[i] = (rhs_r[i] - a[i] * dp[i - 1]) / denom;
        }
        // back-substitute
        out[static_cast<size_t>(N) - 1] = dp[static_cast<size_t>(N) - 1];
        for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
            out[static_cast<size_t>(i)] = dp[static_cast<size_t>(i)] - cp[static_cast<size_t>(i)] * out[static_cast<size_t>(i + 1)];
        }
    };

    // Preconditioning application: left-precondition M^{-1} where rho-block uses
    // the tridiagonal solve and all other variables are scaled by Mdiag.
    const size_t off_rho = static_cast<size_t>(state_vector_size() - N);
    auto apply_left_prec = [&](const Vec& in, Vec& out) {
        out.assign(n, 0.0);
        // scale other variables by Mdiag
        for (size_t i = 0; i < n; ++i) out[i] = in[i] / Mdiag[i];
        // replace rho slice by tridiagonal solve result
        Vec rhs_r(static_cast<size_t>(N), 0.0), sol_r(static_cast<size_t>(N), 0.0);
        // If the caller scaled the full vector by Mdiag earlier, make the rho
        // tridiagonal solve consistent by using the same scaling for the rho
        // slice. This avoids an inconsistent double-scaling where other blocks
        // receive 1/Mdiag but rho was solved with an unscaled rhs.
        for (size_t i = 0; i < N; ++i) rhs_r[i] = in[off_rho + i] / Mdiag[off_rho + i];
        solve_tridiag(rho_a, rho_b, rho_c, rhs_r, sol_r);
        for (size_t i = 0; i < N; ++i) out[off_rho + i] = sol_r[i];
    };

    // Preconditioned residual vector (left preconditioning): r = M^{-1} rhs
    Vec r;
    apply_left_prec(rhs, r);

    // Keep both the preconditioned-residual norm and the original rhs norm.
    // Previously rhs_norm was computed from the preconditioned r which made
    // the early-exit check always compare equal quantities. Compare the
    // preconditioned residual against the unpreconditioned rhs instead.
    const double precond_r_norm = std::max(1e-16, l2_norm(r));
    const double rhs_unpre_norm = std::max(1e-16, l2_norm(rhs));
    double beta = precond_r_norm;
    if (beta / rhs_unpre_norm < p_.linear_tol) {
        return true;
    }

    int total_it = 0;
    while (total_it < max_it) {
        std::vector<Vec> V(static_cast<size_t>(m + 1), Vec(n, 0.0));
        std::vector<Vec> H(static_cast<size_t>(m + 1), Vec(static_cast<size_t>(m), 0.0));
        Vec cs(static_cast<size_t>(m), 0.0);
        Vec sn(static_cast<size_t>(m), 0.0);
        Vec g(static_cast<size_t>(m + 1), 0.0);
        // Record relative residual history inside this restart cycle for
        // diagnostics (useful to inspect stagnation patterns).
        std::vector<double> gmres_rel_history;

        for (size_t i = 0; i < n; ++i) {
            V[0][i] = r[i] / beta;
        }
        g[0] = beta;

        int j_end = 0;
        bool converged = false;

        for (int j = 0; j < m && total_it < max_it; ++j, ++total_it) {
            Vec w_tmp;
            jacobian_vector_product_fd(x_base, r_base, V[static_cast<size_t>(j)], w_tmp);
            // Apply left preconditioning (scaling + rho tridiag solve)
            Vec w(n, 0.0);
            apply_left_prec(w_tmp, w);

            // Modified Gram-Schmidt orthogonalization with one reorthogonalization
            for (int i = 0; i <= j; ++i) {
                double hij = std::inner_product(w.begin(), w.end(), V[static_cast<size_t>(i)].begin(), 0.0);
                H[static_cast<size_t>(i)][static_cast<size_t>(j)] = hij;
                for (size_t k = 0; k < n; ++k) w[k] -= hij * V[static_cast<size_t>(i)][k];
            }
            // Re-orthogonalize (optional one-pass) to improve numerical stability
            for (int i = 0; i <= j; ++i) {
                double corr = std::inner_product(w.begin(), w.end(), V[static_cast<size_t>(i)].begin(), 0.0);
                if (std::abs(corr) > 1e-14) {
                    H[static_cast<size_t>(i)][static_cast<size_t>(j)] += corr;
                    for (size_t k = 0; k < n; ++k) w[k] -= corr * V[static_cast<size_t>(i)][k];
                }
            }

            const double h_next = l2_norm(w);
            H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)] = h_next;
            if (h_next > 1e-18) {
                for (size_t k = 0; k < n; ++k) V[static_cast<size_t>(j + 1)][k] = w[k] / h_next;
            }

            for (int i = 0; i < j; ++i) {
                const double tmp = cs[static_cast<size_t>(i)] * H[static_cast<size_t>(i)][static_cast<size_t>(j)]
                                 + sn[static_cast<size_t>(i)] * H[static_cast<size_t>(i + 1)][static_cast<size_t>(j)];
                H[static_cast<size_t>(i + 1)][static_cast<size_t>(j)] =
                    -sn[static_cast<size_t>(i)] * H[static_cast<size_t>(i)][static_cast<size_t>(j)]
                    + cs[static_cast<size_t>(i)] * H[static_cast<size_t>(i + 1)][static_cast<size_t>(j)];
                H[static_cast<size_t>(i)][static_cast<size_t>(j)] = tmp;
            }

            const double h1 = H[static_cast<size_t>(j)][static_cast<size_t>(j)];
            const double h2 = H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)];
            const double denom = std::sqrt(h1 * h1 + h2 * h2);

            if (denom < 1e-20) {
                cs[static_cast<size_t>(j)] = 1.0;
                sn[static_cast<size_t>(j)] = 0.0;
            } else {
                cs[static_cast<size_t>(j)] = h1 / denom;
                sn[static_cast<size_t>(j)] = h2 / denom;
            }

            H[static_cast<size_t>(j)][static_cast<size_t>(j)] =
                cs[static_cast<size_t>(j)] * h1 + sn[static_cast<size_t>(j)] * h2;
            H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)] = 0.0;

            const double gj = g[static_cast<size_t>(j)];
            g[static_cast<size_t>(j)] = cs[static_cast<size_t>(j)] * gj;
            g[static_cast<size_t>(j + 1)] = -sn[static_cast<size_t>(j)] * gj;

            // Report relative residual for the Arnoldi step normalized by the
            // initial preconditioned residual (so stepwise progress is visible
            // even when the preconditioner rescales the problem).
            const double rel_res = std::abs(g[static_cast<size_t>(j + 1)]) / precond_r_norm;
            // store relative residual at this Arnoldi step
            gmres_rel_history.push_back(rel_res);
            j_end = j + 1;
            if (rel_res < p_.linear_tol) {
                converged = true;
                break;
            }
        }

        Vec y(static_cast<size_t>(j_end), 0.0);
        for (int i = j_end - 1; i >= 0; --i) {
            double s = g[static_cast<size_t>(i)];
            for (int k = i + 1; k < j_end; ++k) {
                s -= H[static_cast<size_t>(i)][static_cast<size_t>(k)] * y[static_cast<size_t>(k)];
            }
            const double diag = H[static_cast<size_t>(i)][static_cast<size_t>(i)];
            y[static_cast<size_t>(i)] = (std::abs(diag) < 1e-20) ? 0.0 : (s / diag);
        }

        for (int j = 0; j < j_end; ++j) {
            const double yj = y[static_cast<size_t>(j)];
            for (size_t k = 0; k < n; ++k) {
                delta[k] += yj * V[static_cast<size_t>(j)][k];
            }
        }

        Vec Ad;
        jacobian_vector_product_fd(x_base, r_base, delta, Ad);
        // Form unpreconditioned residual then apply left preconditioner
        Vec r_unpre(n, 0.0);
        for (size_t i = 0; i < n; ++i) r_unpre[i] = rhs[i] - Ad[i];
        apply_left_prec(r_unpre, r);
        beta = l2_norm(r);

        // Print GMRES residual history for this restart cycle (short summary)
        if (!gmres_rel_history.empty()) {
            double minr = gmres_rel_history.front();
            double maxr = gmres_rel_history.front();
            for (double vv : gmres_rel_history) { if (vv < minr) minr = vv; if (vv > maxr) maxr = vv; }
            std::cout << "[debug-gmres] restart_cycle=" << (total_it / m) << " steps=" << gmres_rel_history.size()
                      << " rel_res_range=[" << minr << "," << maxr << "] final_rel=" << (beta / rhs_unpre_norm) << "\n";
        }

        if (converged || beta / rhs_unpre_norm < p_.linear_tol) {
            return true;
        }
    }

    return false;
}

bool BosonStarSolver::assemble_jacobian(const Vec& x_base, std::vector<double>& J) {
    const size_t n = x_base.size();
    J.assign(n * n, 0.0);

    // Pack base residual
    // Use direct assignment to state_ to avoid clamping in unpack_state_vector
    // like jacobian_vector_product_fd does. We'll copy base x into state_,
    // compute base residual, then perturb each column.
    Vec r_base;
    // apply x_base into state_
    {
        const size_t N = static_cast<size_t>(p_.Nr);
        size_t off = 0;
        for (size_t i = 0; i < N; ++i) state_.metric_f[i] = x_base[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.metric_l[i] = x_base[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.phi[i] = x_base[off + i];
        off += N;
        if (omega_is_unknown()) {
            for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x_base[off + i];
            off += N;
        } else {
            std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
        }
        for (size_t i = 0; i < N; ++i) state_.rho[i] = x_base[off + i];
    }

    const Residuals R0 = compute_residuals();
    pack_residual_vector(R0, r_base);

    // finite-difference eps per-column similar to jacobian_vector_product_fd
    const double nx = std::max(1.0, l2_norm(x_base));
    for (size_t j = 0; j < n; ++j) {
        Vec x_pert = x_base;
        const double xj = std::abs(x_base[j]);
        double eps = p_.jacobian_fd_eps * std::max(1.0, xj) * nx / std::max(1e-16, l2_norm(x_base));
        // simpler: scale with x magnitude
        eps = p_.jacobian_fd_eps * std::max(1.0, xj);
        eps = std::max(1e-12, std::min(1e-3, eps));
        x_pert[j] += eps;

        // assign perturbed x directly into state_
        const size_t N = static_cast<size_t>(p_.Nr);
        size_t off = 0;
        for (size_t i = 0; i < N; ++i) state_.metric_f[i] = x_pert[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.metric_l[i] = x_pert[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.phi[i] = x_pert[off + i];
        off += N;
        if (omega_is_unknown()) {
            for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x_pert[off + i];
            off += N;
        } else {
            std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
        }
        for (size_t i = 0; i < N; ++i) state_.rho[i] = x_pert[off + i];

        const Residuals Rp = compute_residuals();
        Vec rp;
        pack_residual_vector(Rp, rp);

        for (size_t i = 0; i < n; ++i) {
            J[i * n + j] = (rp[i] - r_base[i]) / eps; // row-major
        }
    }

    // restore base state (assign x_base)
    {
        const size_t N = static_cast<size_t>(p_.Nr);
        size_t off = 0;
        for (size_t i = 0; i < N; ++i) state_.metric_f[i] = x_base[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.metric_l[i] = x_base[off + i];
        off += N;
        for (size_t i = 0; i < N; ++i) state_.phi[i] = x_base[off + i];
        off += N;
        if (omega_is_unknown()) {
            for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x_base[off + i];
            off += N;
        } else {
            std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
        }
        for (size_t i = 0; i < N; ++i) state_.rho[i] = x_base[off + i];
    }

    return true;
}

void BosonStarSolver::assign_state_raw(const Vec& x) {
    // Like unpack_state_vector but without clamping, so finite-difference
    // perturbations see a smooth residual.
    const size_t N = static_cast<size_t>(p_.Nr);
    size_t off = 0;
    for (size_t i = 0; i < N; ++i) state_.metric_f[i] = x[off + i];
    off += N;
    for (size_t i = 0; i < N; ++i) state_.metric_l[i] = x[off + i];
    off += N;
    for (size_t i = 0; i < N; ++i) state_.phi[i] = x[off + i];
    off += N;
    if (omega_is_unknown()) {
        for (size_t i = 0; i < N; ++i) state_.omega_s[i] = x[off + i];
        off += N;
    } else {
        std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
    }
    for (size_t i = 0; i < N; ++i) state_.rho[i] = x[off + i];
}

bool BosonStarSolver::assemble_and_solve_banded(const Vec& x_base, const Vec& rhs, Vec& delta) {
    // State/residual vectors are field-major (j = a*N + i). For the solve we
    // interleave fields per grid point (p = i*nf + a), which makes the Jacobian
    // banded: every residual at point k depends only on fields at k-2..k+2.
    const int N = p_.Nr;
    const int nf = static_cast<int>(active_field_count());
    const int n = N * nf;
    const int reach = 2;                 // stencil half-width in grid points
    const int ncolor = 2 * reach + 1;    // columns this far apart touch disjoint rows
    const int kl = (reach + 1) * nf - 1;
    const int ku = kl;
    const int w = band_storage_width(kl, ku);

    std::vector<double> band(static_cast<size_t>(n) * static_cast<size_t>(w), 0.0);
    auto band_at = [&](int row_p, int col_p) -> double& {
        return band[static_cast<size_t>(row_p) * static_cast<size_t>(w) + static_cast<size_t>(col_p - row_p + kl)];
    };

    // Columns that reach beyond the stencil: in lapse continuation the omega
    // boundary row at infinity uses the central lapse sqrt(f(0)).
    std::vector<int> full_cols;
    if (is_lapse_continuation()) full_cols.push_back(0); // field f, point 0
    auto is_full = [&](int j) { return std::find(full_cols.begin(), full_cols.end(), j) != full_cols.end(); };

    // Central differences: the compactified system is ill-conditioned enough at
    // Nr ~ 400 that the ~1e-8 relative error of forward differences spoils Newton
    // steps (e.g. mini-star omega continuation). Step ~ eps_mach^(1/3) balances
    // O(step^2) truncation against round-off, giving ~1e-11 relative accuracy.
    // The step is relative to |x_j|, floored at 1e-3 of the field's typical size
    // (fields pass through zero, e.g. the phi tail). An absolute floor (old
    // max(1, |x_j|)) breaks for compact stars: f(0) ~ 1e-5 is smaller than the step.
    const double step_rel = std::cbrt(std::numeric_limits<double>::epsilon());
    std::vector<double> typ(static_cast<size_t>(nf), 0.0);
    for (int a = 0; a < nf; ++a) {
        for (int i = 0; i < N; ++i)
            typ[static_cast<size_t>(a)] = std::max(typ[static_cast<size_t>(a)], std::abs(x_base[static_cast<size_t>(a * N + i)]));
        if (typ[static_cast<size_t>(a)] == 0.0) typ[static_cast<size_t>(a)] = 1.0;
    }
    auto fd_step = [&](int j) {
        const double xj = std::abs(x_base[static_cast<size_t>(j)]);
        return step_rel * std::max(xj, 1e-3 * typ[static_cast<size_t>(j / N)]);
    };

    Vec xp, xm, rp, rm;
    auto eval_pm = [&]() {
        assign_state_raw(xp);
        pack_residual_vector(compute_residuals(), rp);
        assign_state_raw(xm);
        pack_residual_vector(compute_residuals(), rm);
    };

    for (int a = 0; a < nf; ++a) {
        for (int c = 0; c < ncolor; ++c) {
            xp = x_base;
            xm = x_base;
            std::vector<int> pts;
            for (int i = c; i < N; i += ncolor) {
                const int j = a * N + i;
                if (is_full(j)) continue;
                xp[static_cast<size_t>(j)] += fd_step(j);
                xm[static_cast<size_t>(j)] -= fd_step(j);
                pts.push_back(i);
            }
            if (pts.empty()) continue;
            eval_pm();
            for (int i : pts) {
                const int j = a * N + i;
                const double e2 = xp[static_cast<size_t>(j)] - xm[static_cast<size_t>(j)];
                for (int k = std::max(0, i - reach); k <= std::min(N - 1, i + reach); ++k) {
                    for (int b = 0; b < nf; ++b) {
                        const size_t row = static_cast<size_t>(b * N + k);
                        band_at(k * nf + b, i * nf + a) = (rp[row] - rm[row]) / e2;
                    }
                }
            }
        }
    }

    auto to_p = [&](const Vec& v) {
        Vec out(v.size());
        for (int a = 0; a < nf; ++a)
            for (int i = 0; i < N; ++i) out[static_cast<size_t>(i * nf + a)] = v[static_cast<size_t>(a * N + i)];
        return out;
    };

    // Couplings the band cannot hold, as rank-1 updates J = B + sum_q e_{row_q} v_q^T.
    struct RowUpdate { int row_p; Vec v; };
    std::vector<RowUpdate> updates;
    for (int j : full_cols) {
        const int a = j / N, i = j % N;
        xp = x_base;
        xm = x_base;
        xp[static_cast<size_t>(j)] += fd_step(j);
        xm[static_cast<size_t>(j)] -= fd_step(j);
        const double e2 = xp[static_cast<size_t>(j)] - xm[static_cast<size_t>(j)];
        eval_pm();
        for (int b = 0; b < nf; ++b) {
            for (int k = 0; k < N; ++k) {
                const size_t row = static_cast<size_t>(b * N + k);
                const double val = (rp[row] - rm[row]) / e2;
                if (std::abs(k - i) <= reach) {
                    band_at(k * nf + b, i * nf + a) = val;
                } else if (val != 0.0) {
                    Vec v(static_cast<size_t>(n), 0.0);
                    v[static_cast<size_t>(i * nf + a)] = val;
                    updates.push_back({k * nf + b, std::move(v)});
                }
            }
        }
    }
    assign_state_raw(x_base);

    if (is_arclength_continuation()) {
        // The arclength row (omega_s at infinity) depends on every unknown, so its
        // colored FD entries are polluted. Its exact derivative is the tangent:
        // keep a unit diagonal in the band and add (t - e_diag) as a rank-1 update.
        const int a_omega = 3; // field order f, l, phi, omega_s, rho
        const int rp_arc = (N - 1) * nf + a_omega;
        for (int c = std::max(0, rp_arc - kl); c <= std::min(n - 1, rp_arc + ku); ++c) band_at(rp_arc, c) = 0.0;
        band_at(rp_arc, rp_arc) = 1.0;
        Vec v = to_p(arc_tangent_);
        v[static_cast<size_t>(rp_arc)] -= 1.0;
        updates.push_back({rp_arc, std::move(v)});
    }

    Vec y;
    if (!banded_lu_solve(band, n, kl, ku, to_p(rhs), y)) return false;

    if (!updates.empty()) {
        // Woodbury with U = [e_row_q], V = [v_q]:
        // x = y - Z t, Z = B^{-1} U, (I + V^T Z) t = V^T y.
        const int m = static_cast<int>(updates.size());
        std::vector<Vec> Z(static_cast<size_t>(m));
        for (int q = 0; q < m; ++q) {
            Vec unit(static_cast<size_t>(n), 0.0);
            unit[static_cast<size_t>(updates[static_cast<size_t>(q)].row_p)] = 1.0;
            if (!banded_lu_solve(band, n, kl, ku, unit, Z[static_cast<size_t>(q)])) return false;
        }
        auto dot = [](const Vec& u, const Vec& w) {
            double s = 0.0;
            for (size_t k = 0; k < u.size(); ++k) s += u[k] * w[k];
            return s;
        };
        std::vector<double> S(static_cast<size_t>(m * m), 0.0);
        Vec st(static_cast<size_t>(m)), t;
        for (int q = 0; q < m; ++q) {
            const Vec& vq = updates[static_cast<size_t>(q)].v;
            for (int s2 = 0; s2 < m; ++s2)
                S[static_cast<size_t>(q * m + s2)] = (q == s2 ? 1.0 : 0.0) + dot(vq, Z[static_cast<size_t>(s2)]);
            st[static_cast<size_t>(q)] = dot(vq, y);
        }
        if (!dense_lu_solve(S, st, t, m)) return false;
        for (int q = 0; q < m; ++q)
            for (int p = 0; p < n; ++p) y[static_cast<size_t>(p)] -= Z[static_cast<size_t>(q)][static_cast<size_t>(p)] * t[static_cast<size_t>(q)];
    }

    delta.assign(static_cast<size_t>(n), 0.0);
    for (int a = 0; a < nf; ++a)
        for (int i = 0; i < N; ++i) delta[static_cast<size_t>(a * N + i)] = y[static_cast<size_t>(i * nf + a)];
    return true;
}

bool BosonStarSolver::apply_newton_step_with_line_search(const Vec& x, const Vec& delta, double current_norm) {
    double lambda = 1.0;
    const double c = p_.armijo_c;

    for (int bt = 0; bt < p_.max_line_search_steps; ++bt) {
        Vec xt = x;
        for (size_t i = 0; i < xt.size(); ++i) {
            xt[i] += lambda * delta[i];
        }

        unpack_state_vector(xt);
        const Residuals Rt = compute_residuals();
        Vec rt;
        pack_residual_vector(Rt, rt);
        const double nt = inf_norm(rt);

        // Require the Armijo sufficient-decrease condition. Previously a
        // permissive 'any-reduction' check allowed tiny inconsistent steps
        // to be accepted; enforcing Armijo improves globalization behavior.
        if (nt <= (1.0 - c * lambda) * current_norm) {
            // Enforce symmetric boundary values after accepting the step so
            // the in-memory state remains regular for subsequent operations
            // and for diagnostic outputs.
            enforce_boundary_values();
            return true;
        }

        lambda *= 0.5;
        if (lambda < p_.line_search_min) {
            break;
        }
    }

    // restore old state if no step accepted
    unpack_state_vector(x);
    return false;
}

double BosonStarSolver::apply_residual_smoother_step(const Residuals& R, double alpha) {
    const int N = p_.Nr;
    const double h = (p_.rbar_max - p_.rbar_min) / static_cast<double>(N - 1);
    const double scale2 = h * h;

    double max_update = 0.0;
    for (int i = 1; i < N - 1; ++i) {
        const size_t k = static_cast<size_t>(i);

        const double df = -alpha * scale2 * R.f[k];
        const double dl = -alpha * scale2 * R.l[k];
        const double dp = -alpha * scale2 * R.phi[k];
        const double drho = -alpha * scale2 * R.rho[k];

        state_.metric_f[k] = std::max(1e-8, state_.metric_f[k] + df);
        state_.metric_l[k] = std::max(1e-8, state_.metric_l[k] + dl);
        state_.phi[k] += dp;
        state_.rho[k] += drho;

        if (omega_is_unknown()) {
            const double dw = -alpha * scale2 * R.omega[k];
            state_.omega_s[k] += dw;
            max_update = std::max(max_update, std::abs(dw));
        }

        max_update = std::max(max_update, std::abs(df));
        max_update = std::max(max_update, std::abs(dl));
        max_update = std::max(max_update, std::abs(dp));
        max_update = std::max(max_update, std::abs(drho));
    }

    // Enforce boundary conditions after the smoother update so the state
    // retains regular origin/infinity values.
    enforce_boundary_values();
    return max_update;
}

void BosonStarSolver::enforce_boundary_values() {
    const int N = p_.Nr;
    if (N < 2) {
        return;
    }
    // Infinity conditions
    state_.metric_f[static_cast<size_t>(N - 1)] = 1.0;
    state_.metric_l[static_cast<size_t>(N - 1)] = 1.0;
    state_.phi[static_cast<size_t>(N - 1)] = 0.0;

    // In lapse continuation, rho is not fixed at infinity (natural BC via
    // residual), so do not hard-clamp rho(N-1) here.
    if (!is_lapse_continuation()) {
        state_.rho[static_cast<size_t>(N - 1)] = 1.0;
    }

    if (is_omega_continuation()) {
        std::fill(state_.omega_s.begin(), state_.omega_s.end(), p_.omega_target);
    } else if (is_lapse_continuation() || is_arclength_continuation()) {
        // omega is solved as an auxiliary field and is not fixed at infinity;
        // keep current values unchanged.
        (void)0;
    } else {
        state_.omega_s[static_cast<size_t>(N - 1)] = state_.omega_s[static_cast<size_t>(N - 2)];
    }
}

bool BosonStarSolver::is_charge_continuation() const {
    return p_.continuation_mode == "charge";
}

bool BosonStarSolver::is_omega_continuation() const {
    // In explicit omega mode we always enforce omega-target boundaries.
    if (p_.continuation_mode == "omega") return true;

    // For continuation_mode='lapse' / 'arclength', keep omega-style enforcement
    // until the dedicated continuation loop is entered (e.g. during the kappa ramp).
    if (p_.continuation_mode == "lapse") {
        if (!lapse_mode_active_) return true;
    }
    if (p_.continuation_mode == "arclength") {
        if (!arclength_active_) return true;
    }
    return false;
}

bool BosonStarSolver::is_lapse_continuation() const {
    return (p_.continuation_mode == "lapse" && lapse_mode_active_);
}

bool BosonStarSolver::is_arclength_continuation() const {
    return (p_.continuation_mode == "arclength" && arclength_active_);
}

bool BosonStarSolver::omega_is_unknown() const {
    return is_charge_continuation() || is_lapse_continuation() || is_arclength_continuation();
}

double BosonStarSolver::arclength_residual(double* scale) const {
    // Pseudo-arclength constraint <t, x - x_pred> = 0 on the full packed state
    // (omega_s included); its derivative with respect to x is exactly t.
    Vec x;
    pack_state_vector(x);
    double s = 0.0, sa = 0.0;
    for (size_t j = 0; j < x.size(); ++j) {
        s += arc_tangent_[j] * (x[j] - arc_pred_[j]);
        sa += std::abs(arc_tangent_[j]) * (std::abs(x[j]) + std::abs(arc_pred_[j]));
    }
    if (scale) *scale = sa;
    return s;
}

bool BosonStarSolver::potential_supports_qball_bootstrap() const {
    return potential_has_qballs(p_);
}

std::pair<double, double> BosonStarSolver::qball_rhs(double r, double phi, double dphi, double omega) const {
    const double dU = dU_dphi(phi);
    const double phi_dd = 0.5 * dU - omega * omega * phi - (2.0 / std::max(1e-12, r)) * dphi;
    return {dphi, phi_dd};
}

bool BosonStarSolver::integrate_qball_profile(double phi0, double omega, QBallProfile& prof, double& phi_end, bool& crossed_zero) const {
    const double dr = p_.qball_dr;
    const double rmax = p_.qball_rmax;
    if (dr <= 0.0 || rmax <= dr) {
        return false;
    }

    const int N = static_cast<int>(std::floor(rmax / dr)) + 1;
    prof.r.assign(static_cast<size_t>(N), 0.0);
    prof.phi.assign(static_cast<size_t>(N), 0.0);
    prof.dphi.assign(static_cast<size_t>(N), 0.0);
    crossed_zero = false;

    const double r0 = dr;
    const double phi2 = 0.5 * dU_dphi(phi0) - omega * omega * phi0;
    double r = r0;
    double phi = phi0 + 0.5 * phi2 * r0 * r0;
    double dphi = phi2 * r0;

    prof.r[0] = 0.0;
    prof.phi[0] = phi0;
    prof.dphi[0] = 0.0;

    for (int i = 1; i < N; ++i) {
        prof.r[static_cast<size_t>(i)] = r;
        prof.phi[static_cast<size_t>(i)] = phi;
        prof.dphi[static_cast<size_t>(i)] = dphi;

        auto [k1_phi, k1_dphi] = qball_rhs(r, phi, dphi, omega);
        auto [k2_phi, k2_dphi] = qball_rhs(r + 0.5 * dr, phi + 0.5 * dr * k1_phi, dphi + 0.5 * dr * k1_dphi, omega);
        auto [k3_phi, k3_dphi] = qball_rhs(r + 0.5 * dr, phi + 0.5 * dr * k2_phi, dphi + 0.5 * dr * k2_dphi, omega);
        auto [k4_phi, k4_dphi] = qball_rhs(r + dr, phi + dr * k3_phi, dphi + dr * k3_dphi, omega);

        phi += (dr / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi);
        dphi += (dr / 6.0) * (k1_dphi + 2.0 * k2_dphi + 2.0 * k3_dphi + k4_dphi);
        r += dr;

        if (phi <= 0.0) {
            crossed_zero = true;
            phi_end = phi;
            prof.r.resize(static_cast<size_t>(i + 1));
            prof.phi.resize(static_cast<size_t>(i + 1));
            prof.dphi.resize(static_cast<size_t>(i + 1));
            return true;
        }

        if (!std::isfinite(phi) || !std::isfinite(dphi) || std::abs(phi) > 1e4) {
            phi_end = phi;
            return false;
        }
    }

    phi_end = prof.phi.back();
    return true;
}
bool BosonStarSolver::find_qball_central_amplitude(double omega, double& phi0_best, QBallProfile& best_profile) const {
    const int scans = std::max(10, p_.qball_scan_points);
    const double a = p_.qball_phi0_min;
    const double b = p_.qball_phi0_max;
    if (a <= 0.0 || b <= a) {
        return false;
    }

    bool have_bracket = false;
    double left = a;
    double right = b;
    double f_left = 0.0;

    double prev_phi0 = a;
    double prev_val = 0.0;
    bool prev_ok = false;

    for (int i = 0; i < scans; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(scans - 1);
        const double phi0 = a + (b - a) * t;

        QBallProfile prof;
        double val = 0.0;
        bool crossed_zero = false;
        const bool ok = integrate_qball_profile(phi0, omega, prof, val, crossed_zero);
        if (!ok) continue;

        const double F = crossed_zero ? -1.0 : val;

        if (prev_ok && (F == 0.0 || (F > 0.0) != (prev_val > 0.0))) {
            left = prev_phi0;
            right = phi0;
            f_left = prev_val;
            have_bracket = true;
            break;
        }

        prev_ok = true;
        prev_phi0 = phi0;
        prev_val = F;
    }

    if (!have_bracket) {
        return false;
    }

    for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (left + right);
        QBallProfile prof_mid;
        double f_mid = 0.0;
        bool crossed_zero = false;
        const bool ok = integrate_qball_profile(mid, omega, prof_mid, f_mid, crossed_zero);
        if (!ok) {
            return false;
        }

        const double Fmid = crossed_zero ? -1.0 : f_mid;

        if (!crossed_zero && std::abs(f_mid) < 1e-8) {
            phi0_best = mid;
            best_profile = std::move(prof_mid);
            return true;
        }

        if ((Fmid > 0.0) == (f_left > 0.0)) {
            left = mid;
            f_left = Fmid;
        } else {
            right = mid;
        }
    }

    // If tolerance wasn't reached in-loop, choose the best non-crossing candidate
    // among the bracketing endpoints and midpoint by minimizing phi_end.
    const double cands[3] = {left, 0.5 * (left + right), right};
    bool have_best = false;
    double best_phi_end = std::numeric_limits<double>::infinity();
    QBallProfile best_prof_local;
    double best_phi0_local = cands[1];

    for (double c : cands) {
        QBallProfile prof_c;
        double phi_end_c = 0.0;
        bool crossed_zero_c = false;
        const bool ok_c = integrate_qball_profile(c, omega, prof_c, phi_end_c, crossed_zero_c);
        if (!ok_c || crossed_zero_c) {
            continue;
        }
        if (phi_end_c < best_phi_end) {
            best_phi_end = phi_end_c;
            best_prof_local = std::move(prof_c);
            best_phi0_local = c;
            have_best = true;
        }
    }

    if (!have_best) {
        return false;
    }

    phi0_best = best_phi0_local;
    best_profile = std::move(best_prof_local);
    return true;
}

double BosonStarSolver::interp_linear(const std::vector<double>& x, const std::vector<double>& y, double xi) const {
    if (x.empty()) return 0.0;
    if (xi <= x.front()) return y.front();
    if (xi >= x.back()) return y.back();

    auto it = std::lower_bound(x.begin(), x.end(), xi);
    size_t j = static_cast<size_t>(it - x.begin());
    if (j == 0) return y[0];
    size_t i = j - 1;
    const double x0 = x[i], x1 = x[j];
    const double t = (xi - x0) / std::max(1e-14, x1 - x0);
    return (1.0 - t) * y[i] + t * y[j];
}

bool BosonStarSolver::bootstrap_from_qball_shooting() {
    const double omega = p_.omega_target;
    const double omega_max = boson_mass(p_);
    if (!(omega > 0.0 && omega < omega_max)) {
        std::cout << "[warn] omega_target outside expected Q-ball shooting window (0, mu).\n";
    }

    double phi0 = 0.0;
    QBallProfile prof;
    const bool ok = find_qball_central_amplitude(omega, phi0, prof);
    if (!ok) {
        return false;
    }

    std::cout << "[info] Q-ball shooting succeeded: phi(0) = " << phi0
              << ", profile points = " << prof.r.size() << "\n";

    if (apply_exponential_tail_to_qball_profile(prof, omega)) {
        std::cout << "[info] Applied in-data exponential tail fit to Q-ball profile (used for seed/output).\n";
    } else {
        std::cout << "[warn] Could not apply in-data exponential tail fit; using raw shooting profile.\n";
    }

    if (!write_qball_dat(prof, omega)) {
        std::cout << "[warn] Failed to write qball.dat diagnostics file.\n";
    } else {
        std::cout << "[info] Wrote Q-ball diagnostics to qball.dat\n";
    }

    const size_t N = state_.rbar.size();
    state_.phi.assign(N, 0.0);
    state_.metric_f.assign(N, 1.0);
    state_.metric_l.assign(N, 1.0);
    state_.rho.assign(N, 1.0);
    state_.omega_s.assign(N, omega);

    for (size_t i = 0; i < N; ++i) {
        const double ri = state_.r[i];
        const double phi_i = interp_linear(prof.r, prof.phi, ri);
        state_.phi[i] = std::max(0.0, phi_i);
        // Use a smooth profile with zero derivative at origin to avoid
        // large 2/r * rho_r terms in the residual near r->0.
        state_.rho[i] = 1.0 - 0.1 * std::exp(-ri * ri);
    }

    enforce_boundary_values();
    return true;
}

bool BosonStarSolver::apply_exponential_tail_to_qball_profile(QBallProfile& prof, double omega) const {
    if (prof.r.size() < 8 || prof.phi.size() != prof.r.size() || prof.dphi.size() != prof.r.size()) {
        return false;
    }

    // Match point based on minimum |phi|, then move inward by r_match_fac.
    int i_min = 0;
    double phi_min_abs = std::abs(prof.phi[0]);
    for (size_t i = 1; i < prof.r.size(); ++i) {
        const double pabs = std::abs(prof.phi[i]);
        if (pabs < phi_min_abs) {
            phi_min_abs = pabs;
            i_min = static_cast<int>(i);
        }
    }

    const double fac = std::max(0.2, std::min(0.99, p_.r_match_fac));
    const double r_min = prof.r[static_cast<size_t>(i_min)];
    const double r_match_target = fac * r_min;

    int i_match = 0;
    for (int i = 0; i <= i_min; ++i) {
        if (prof.r[static_cast<size_t>(i)] >= r_match_target) {
            i_match = i;
            break;
        }
    }
    i_match = std::max(0, std::min(static_cast<int>(prof.r.size()) - 2, i_match));

    const double mu = std::max(1e-12, boson_mass(p_));
    const double beta = std::sqrt(std::max(1e-12, mu * mu - omega * omega));
    const double r_match = std::max(1e-8, prof.r[static_cast<size_t>(i_match)]);

    // Continuity of phi at match point sets A.
    const double phi_match = std::max(1e-20, std::abs(prof.phi[static_cast<size_t>(i_match)]));
    const double A = phi_match * r_match * std::exp(beta * r_match);

    // Overwrite tail in profile data itself (this is used as seed and output).
    for (size_t i = static_cast<size_t>(i_match); i < prof.r.size(); ++i) {
        const double r = std::max(1e-8, prof.r[i]);
        const double phi_fit = A * std::exp(-beta * r) / r;
        const double dphi_fit = -phi_fit * (beta + 1.0 / r);
        prof.phi[i] = phi_fit;
        prof.dphi[i] = dphi_fit;
    }

    return true;
}

bool BosonStarSolver::write_qball_dat(const QBallProfile& prof, double omega) const {
    if (prof.r.empty() || prof.phi.size() != prof.r.size() || prof.dphi.size() != prof.r.size()) {
        return false;
    }

    std::ofstream out("qball.dat");
    if (!out.is_open()) {
        return false;
    }

    out << "# Q-ball profile diagnostics (flat-space shooting stage)\n";
    out << "# potential=" << p_.potential << " omega=" << omega << " mu=" << p_.mu
        << " kappa=0" << "\n";
    out << "# columns:\n";
    out << "# r phi dphi U grad_energy time_energy energy_density j_t shell_mass_density shell_charge_density phi_fit dphi_fit\n";

    // Build a diagnostic fitted tail model using the requested matching recipe:
    // choose where |phi| is minimal, then move inward by r_match_fac.
    int i_min = 0;
    double phi_min_abs = std::abs(prof.phi[0]);
    for (size_t i = 1; i < prof.r.size(); ++i) {
        const double pabs = std::abs(prof.phi[i]);
        if (pabs < phi_min_abs) {
            phi_min_abs = pabs;
            i_min = static_cast<int>(i);
        }
    }

    const double fac = std::max(0.2, std::min(0.99, p_.r_match_fac));
    const double r_min = prof.r[static_cast<size_t>(i_min)];
    const double r_match_target = fac * r_min;
    int i_match = 0;
    for (int i = 0; i <= i_min; ++i) {
        if (prof.r[static_cast<size_t>(i)] >= r_match_target) {
            i_match = i;
            break;
        }
    }

    const double mu = std::max(1e-12, boson_mass(p_));
    const double beta = std::sqrt(std::max(1e-12, mu * mu - omega * omega));
    const double r_match = std::max(1e-8, prof.r[static_cast<size_t>(i_match)]);
    const double phi_match = std::max(1e-20, std::abs(prof.phi[static_cast<size_t>(i_match)]));
    const double A = phi_match * r_match * std::exp(beta * r_match);

    out << std::scientific << std::setprecision(12);
    for (size_t i = 0; i < prof.r.size(); ++i) {
        const double r = prof.r[i];
        const double phi = prof.phi[i];
        const double dphi = prof.dphi[i];

        const double U = potential_U(phi);
        const double grad_energy = dphi * dphi;
        const double time_energy = omega * omega * phi * phi;
        const double energy_density = time_energy + grad_energy + U;

        // Noether-charge density for harmonic ansatz in flat space.
        const double j_t = 2.0 * omega * phi * phi;

        // Radial shell integrands (so M = ∫ shell_mass_density dr, Q = ∫ shell_charge_density dr).
        const double r2 = r * r;
        const double shell_mass_density = 4.0 * kPi * r2 * energy_density;
        const double shell_charge_density = 8.0 * kPi * omega * r2 * phi * phi;

        const double phi_fit = (r > 0.0) ? (A * std::exp(-beta * r) / r) : phi_match;
        const double dphi_fit = (r > 0.0) ? (-phi_fit * (beta + 1.0 / r)) : 0.0;

        out << r << ' '
            << phi << ' '
            << dphi << ' '
            << U << ' '
            << grad_energy << ' '
            << time_energy << ' '
            << energy_density << ' '
            << j_t << ' '
            << shell_mass_density << ' '
        << shell_charge_density << ' '
        << phi_fit << ' '
        << dphi_fit << '\n';
    }

    return out.good();
}

bool BosonStarSolver::write_bs_dat() {
    // Ensure symmetric/regular boundary at origin before writing.
    enforce_boundary_values();

    const size_t N = state_.r.size();
    if (N == 0) return false;

    std::ofstream out("bs.dat");
    if (!out.is_open()) return false;

    out << "# Boson-star profile diagnostics\n";
    out << "# potential=" << p_.potential << " kappa=" << current_kappa_ << " mu=" << p_.mu << "\n";
    out << "# columns: r phi metric_f metric_l\n";

    out << std::scientific << std::setprecision(12);
    for (size_t i = 0; i < N - 1; ++i) {
        out << state_.r[i] << ' '
            << state_.phi[i] << ' '
            << state_.metric_f[i] << ' '
            << state_.metric_l[i] << '\n';
    }

    return out.good();
}

bool BosonStarSolver::append_family_model(const std::string& filename, bool overwrite, double omega) {
    // Ensure symmetric/regular boundary at origin before writing.
    enforce_boundary_values();

    const size_t N = state_.r.size();
    if (N == 0) return false;

    std::ofstream out;
    if (overwrite) out.open(filename);
    else out.open(filename, std::ios::app);
    if (!out.is_open()) return false;

    // Write a single-line diagnostics entry per model (one BS per line).
    // Columns mirror `initial_models.dat`: kappa omega max_phi mass charge residual_inf
    if (overwrite) {
        out << "# Families of solutions - one line per model:\n";
        out << "# kappa omega max_phi mass charge lapse_center residual_inf mass_komar charge_integral\n";
    }

    // Compute max |phi|
    double max_phi = 0.0;
    for (double v : state_.phi) max_phi = std::max(max_phi, std::abs(v));

    const double mass = compute_current_mass();
    const double charge = compute_current_charge();

    // central lapse: sqrt(metric_f at origin)
    const double lapse_c = std::sqrt(std::max(1e-16, state_.metric_f[0]));
    out << std::scientific << std::setprecision(12)
        << current_kappa_ << ' ' << omega << ' '
        << max_phi << ' ' << mass << ' ' << charge << ' ' << lapse_c << ' ' << last_residual_norm_
        << ' ' << compute_komar_mass() << ' ' << compute_charge_integral() << '\n';

    return out.good();
}

double BosonStarSolver::compute_current_charge() const {
    const size_t N = state_.r.size();
    if (N < 3) return 0.0;
    const int iAsym = static_cast<int>(N) - 2;
    const double rAsym = std::max(1e-8, state_.r[static_cast<size_t>(iAsym)]);
    const double rbAsym = state_.rbar[static_cast<size_t>(iAsym)];
    const double h = (p_.rbar_max - p_.rbar_min) / static_cast<double>(N - 1);

    const double rho_b_asym = d1_rbar(state_.rho, iAsym, h);
    const double rho_r_asym = d1_r_from_rbar(rho_b_asym, rbAsym);
    const double w_asym = state_.omega_s[static_cast<size_t>(iAsym)];
    const double charge = 8.0 * kPi * rAsym * rAsym * w_asym * rho_r_asym;
    return charge;
}

double BosonStarSolver::compute_current_mass() const {
    // Outside the star the metric is isotropic Schwarzschild,
    //   f = ((1 - M/2r) / (1 + M/2r))^2  =>  M = 2 r (1 - sqrt f) / (1 + sqrt f),
    // which is exact at any exterior radius (no O(M/r) truncation). Units G = 1,
    // i.e. kappa = 8 pi; for other kappa this is M*G with G = kappa / (8 pi).
    const size_t N = state_.r.size();
    if (N < 3) return 0.0;
    const size_t iAsym = N - 2;
    const double rAsym = state_.r[iAsym];
    const double sf = std::sqrt(std::max(0.0, state_.metric_f[iAsym]));
    const double MG = 2.0 * rAsym * (1.0 - sf) / (1.0 + sf);
    // f only knows M*G, with G = kappa / (8 pi). In flat space (kappa = 0) use the
    // Komar integral, which needs no G.
    if (current_kappa_ <= 0.0) return compute_komar_mass();
    return MG * 8.0 * kPi / current_kappa_;
}

double BosonStarSolver::radial_integral(const std::vector<double>& integrand_r) const {
    // Integral over r in [0, inf) of a function sampled on the grid: dr = drbar / (1 - rbar)^2,
    // composite Simpson in rbar (trapezoid on a leftover interval). The integrand must decay
    // at infinity; the last node (r = inf) is taken as zero.
    const size_t N = state_.r.size();
    const double h = (p_.rbar_max - p_.rbar_min) / static_cast<double>(N - 1);
    std::vector<double> g(N, 0.0);
    for (size_t i = 0; i + 1 < N; ++i) {
        const double om = 1.0 - state_.rbar[i];
        g[i] = integrand_r[i] / (om * om);
    }
    double s = 0.0;
    size_t i = 0;
    for (; i + 2 < N; i += 2) s += (h / 3.0) * (g[i] + 4.0 * g[i + 1] + g[i + 2]);
    if (i + 1 < N) s += 0.5 * h * (g[i] + g[i + 1]);
    return s;
}

double BosonStarSolver::compute_komar_mass() const {
    // Komar/Tolman mass M = -int (2 T_t^t - T) sqrt(-g) d^3x (Kleihaus et al. Eq. 20), with
    // sqrt(-g) = l^{3/2} r^2 sin(theta) / f and the App. C stress tensor:
    //   2 T_t^t - T = 2 U - 4 omega^2 phi^2 / f.
    // Equals the asymptotic mass for exact solutions; in flat space it is the Q-ball energy.
    const size_t N = state_.r.size();
    std::vector<double> integ(N, 0.0);
    for (size_t i = 0; i + 1 < N; ++i) {
        const double f = state_.metric_f[i], l = state_.metric_l[i], phi = state_.phi[i];
        const double w = state_.omega_s[i], r = state_.r[i];
        integ[i] = (4.0 * w * w * phi * phi / f - 2.0 * potential_U(phi)) * std::pow(l, 1.5) * r * r / f;
    }
    return 4.0 * kPi * radial_integral(integ);
}

double BosonStarSolver::compute_charge_integral() const {
    // Q = 4 pi omega int int sqrt(-g) phi^2 / f dr dtheta (Kleihaus et al. Eq. 23, n = 0)
    //   = 8 pi omega int l^{3/2} r^2 phi^2 / f^2 dr.
    const size_t N = state_.r.size();
    std::vector<double> integ(N, 0.0);
    for (size_t i = 0; i + 1 < N; ++i) {
        const double f = state_.metric_f[i], l = state_.metric_l[i], phi = state_.phi[i], r = state_.r[i];
        integ[i] = std::pow(l, 1.5) * r * r * phi * phi / (f * f);
    }
    return 8.0 * kPi * state_.omega_s[N - 2] * radial_integral(integ);
}

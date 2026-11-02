#include "rotating_solver.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <limits>

#include "linear_solvers.h"
#include "potentials.h"

namespace {
constexpr double kPi = 3.1415926535897932384626433832795;

// Composite Simpson weights on n uniformly spaced points (trapezoid on a leftover interval).
std::vector<double> simpson_weights(int n, double h) {
    std::vector<double> w(static_cast<size_t>(n), 0.0);
    int i = 0;
    for (; i + 2 < n; i += 2) {
        w[static_cast<size_t>(i)] += h / 3.0;
        w[static_cast<size_t>(i + 1)] += 4.0 * h / 3.0;
        w[static_cast<size_t>(i + 2)] += h / 3.0;
    }
    if (i + 1 < n) {
        w[static_cast<size_t>(i)] += 0.5 * h;
        w[static_cast<size_t>(i + 1)] += 0.5 * h;
    }
    return w;
}

// log cosh z without overflow
double logcosh(double z) {
    const double a = std::abs(z);
    return a + std::log1p(std::exp(-2.0 * a)) - std::log(2.0);
}
}

double RadialMap::Cluster::B(double r) const {
    return 0.5 * delta * (logcosh((r - ra) / delta) - logcosh(ra / delta) - logcosh((r - rb) / delta) + logcosh(rb / delta));
}

double RadialMap::Cluster::B_inf() const {
    return 0.5 * ((rb - ra) - delta * logcosh(ra / delta) + delta * logcosh(rb / delta));
}

double RadialMap::xi(double r) const {
    const double bs = beta_sum();
    if (bs == 0.0) return r / (L + r);
    double v = r / (L + r);
    for (const Cluster& c : cl)
        if (c.beta > 0.0) v += c.beta * c.B(r) / c.B_inf();
    return v / (1.0 + bs);
}

void RadialMap::derivs(double r, double& xi_r, double& xi_rr) const {
    const double s = 1.0 / (L + r);
    xi_r = L * s * s;
    xi_rr = -2.0 * L * s * s * s;
    const double bs = beta_sum();
    if (bs == 0.0) return;
    for (const Cluster& c : cl) {
        if (c.beta <= 0.0) continue;
        const double ta = std::tanh((r - c.ra) / c.delta), tb = std::tanh((r - c.rb) / c.delta);
        const double w = c.beta / c.B_inf();
        xi_r += w * 0.5 * (ta - tb);
        xi_rr += w * 0.5 * ((1.0 - ta * ta) - (1.0 - tb * tb)) / c.delta;
    }
    xi_r /= 1.0 + bs;
    xi_rr /= 1.0 + bs;
}

double RadialMap::r_of_xi(double x) const {
    if (beta_sum() == 0.0) return L * x / (1.0 - x);
    // xi is increasing: bracket, then safeguarded Newton.
    double lo = 0.0, hi = L;
    while (xi(hi) < x) { lo = hi; hi *= 2.0; }
    double r = 0.5 * (lo + hi);
    for (int it = 0; it < 200; ++it) {
        const double f = xi(r) - x;
        if (f > 0.0) hi = r; else lo = r;
        double d, dd;
        derivs(r, d, dd);
        double rn = r - f / d;
        if (!(rn > lo && rn < hi)) rn = 0.5 * (lo + hi);
        if (std::abs(rn - r) <= 1e-15 * std::max(1.0, r)) { r = rn; break; }
        r = rn;
    }
    return r;
}

Ergoregion find_ergoregion(const std::vector<double>& r, const std::vector<double>& f, const std::vector<double>& l,
                           const std::vector<double>& W, double gtt_max) {
    Ergoregion out;
    out.gtt_max = gtt_max;
    const size_t n = r.size();
    auto gtt = [&](size_t k) { return -f[k] + l[k] * W[k] * W[k] / f[k]; };
    for (size_t k = 0; k + 1 < n; ++k) {
        const double a = gtt(k), b = gtt(k + 1);
        if ((a > 0.0) == (b > 0.0)) continue;
        const double t = a / (a - b);
        const double Ra = r[k] * std::sqrt(l[k] / f[k]), Rb = r[k + 1] * std::sqrt(l[k + 1] / f[k + 1]);
        out.R.push_back(Ra + t * (Rb - Ra));
    }
    return out;
}

LightRings find_light_rings(const std::vector<double>& r, const std::vector<double>& f, const std::vector<double>& l,
                            const std::vector<double>& W) {
    LightRings out;
    const size_t n = r.size();
    if (n < 3) return out;
    // Sense of rotation of the frame dragging (W/r), read where it is largest in magnitude.
    double Wmax = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (std::abs(W[i]) > std::abs(Wmax)) Wmax = W[i];
    std::vector<double> H(n), Rc(n);
    for (size_t i = 0; i < n; ++i) Rc[i] = r[i] * std::sqrt(l[i] / f[i]);
    for (int s : {+1, -1}) {
        for (size_t i = 0; i < n; ++i) H[i] = W[i] / r[i] + s * f[i] / (r[i] * std::sqrt(l[i]));
        const bool co = (Wmax == 0.0) ? (s > 0) : ((s > 0) == (Wmax > 0.0));
        if (Wmax == 0.0 && s < 0) break;  // static: H_- = -H_+, same rings
        for (size_t i = 1; i + 1 < n; ++i) {
            const double d0 = H[i] - H[i - 1], d1 = H[i + 1] - H[i];
            if (!(d0 * d1 < 0.0)) continue;
            const double den = H[i - 1] - 2.0 * H[i] + H[i + 1];
            double u = (den != 0.0) ? 0.5 * (H[i - 1] - H[i + 1]) / den : 0.0;  // offset in [-1/2, 1/2]
            u = std::max(-0.5, std::min(0.5, u));
            const size_t k = (u >= 0.0) ? i : i - 1;
            const double t = (u >= 0.0) ? u : 1.0 + u;
            const double R = (1.0 - t) * Rc[k] + t * Rc[k + 1];
            (co ? out.R_co : out.R_counter).push_back(R);
        }
    }
    return out;
}

RotatingSolver::RotatingSolver(const BSParams& params)
    : p_(params), Nr_(params.Nr), Nt_(params.Ntheta), m_(params.m_azimuthal), hr_(0), ht_(0) {}

void RotatingSolver::build_grid() {
    hr_ = (p_.rbar_max - p_.rbar_min) / static_cast<double>(Nr_ - 1);
    ht_ = (theta_full_ ? kPi : 0.5 * kPi) / static_cast<double>(Nt_ - 1);
    je_ = theta_full_ ? (Nt_ - 1) / 2 : Nt_ - 1;
    rbar_.resize(static_cast<size_t>(Nr_));
    r_.resize(static_cast<size_t>(Nr_));
    xi_r_.assign(static_cast<size_t>(Nr_), 0.0);
    xi_rr_.assign(static_cast<size_t>(Nr_), 0.0);
    theta_.resize(static_cast<size_t>(Nt_));
    for (int i = 0; i < Nr_; ++i) {
        const size_t k = static_cast<size_t>(i);
        rbar_[k] = p_.rbar_min + hr_ * i;
        if (i == Nr_ - 1) { r_[k] = 1.0e12; continue; }
        r_[k] = map_.r_of_xi(rbar_[k]);
        map_.derivs(r_[k], xi_r_[k], xi_rr_[k]);
    }
    for (int j = 0; j < Nt_; ++j) theta_[static_cast<size_t>(j)] = ht_ * j;

    // Quadrature over dr dtheta dvarphi on both hemispheres: 2 pi * 2 * (r-weights) * (theta-weights),
    // dr = dxi / xi'(r); the node at r = infinity carries no weight.
    const std::vector<double> wr = simpson_weights(Nr_, hr_);
    const std::vector<double> wt = simpson_weights(Nt_, ht_);
    wq_th_.resize(static_cast<size_t>(Nt_));
    // 2 pi from varphi, and a factor 2 for the mirrored hemisphere unless theta covers [0, pi]
    const double hemis = theta_full_ ? 2.0 * kPi : 4.0 * kPi;
    for (int j = 0; j < Nt_; ++j) wq_th_[static_cast<size_t>(j)] = hemis * wt[static_cast<size_t>(j)];
    wq_.assign(static_cast<size_t>(Nr_) * Nt_, 0.0);
    for (int i = 0; i + 1 < Nr_; ++i) {
        for (int j = 0; j < Nt_; ++j)
            wq_[static_cast<size_t>(i) * Nt_ + j] =
                wq_th_[static_cast<size_t>(j)] * wr[static_cast<size_t>(i)] / xi_r_[static_cast<size_t>(i)];
    }
}

RadialMap RotatingSolver::wanted_map(const Vec& x) const {
    // On every theta ray carrying at least a quarter of max|phi|: the outer wall (the steepest
    // descent of |phi| in ln r beyond the ray's peak, max of -r d|phi|/dr, at r_w; thickness
    // t = 2 |phi(r_w)| / slope there: 2 delta for a tanh wall, twice the decay length for a diffuse
    // star; weighting by r keeps a steep central spike, which can exceed twice the plateau deep in
    // the spiral, from being taken for the wall) and the inner rise (the first radius r_i
    // where |phi| reaches half the peak, thickness t_i = peak / slope there; a spike in the core of
    // strongly relativistic stars). Half crossings are interpolated, so the map moves smoothly
    // along a family instead of jumping with the steepest grid slope.
    // Wall cluster: [min(r_w - t), max(r_w + t)]; core cluster: |r| <= max(r_i + t_i). The cluster
    // fraction is shared in proportion to the number of thicknesses each cluster spans, with at
    // least 16 points per cluster.
    double P = 0.0;
    for (int i = 0; i + 1 < Nr_; ++i)
        for (int j = 0; j < Nt_; ++j) P = std::max(P, amp(x, i, j));
    const double inf = std::numeric_limits<double>::infinity();
    double lo = inf, hi = 0.0, tmin = inf, core = 0.0, tcore = inf;
    for (int j = 0; j < Nt_; ++j) {
        auto a = [&](int i) { return amp(x, i, j); };
        auto slope = [&](int i) { return (a(i + 1) - a(i - 1)) / (r_[static_cast<size_t>(i + 1)] - r_[static_cast<size_t>(i - 1)]); };
        int pk = 0;
        for (int i = 1; i + 1 < Nr_; ++i) if (a(i) > a(pk)) pk = i;
        if (a(pk) < 0.25 * P) continue;
        double best = 0.0;
        int iw = -1;
        for (int i = std::max(pk, 1); i + 2 < Nr_; ++i) {
            const double v = -r_[static_cast<size_t>(i)] * slope(i);
            if (v > best) { best = v; iw = i; }
        }
        if (iw >= 0) {
            const double t = 2.0 * a(iw) / (-slope(iw)), rw = r_[static_cast<size_t>(iw)];
            lo = std::min(lo, rw - t);
            hi = std::max(hi, rw + t);
            tmin = std::min(tmin, t);
        }
        int ii = 1;
        while (ii < pk && a(ii) < 0.5 * a(pk)) ++ii;  // a(ii-1) < peak/2 <= a(ii)
        if (pk > 0 && ii <= pk) {
            const double s = std::max(slope(std::max(ii - 1, 1)), slope(ii));
            if (s > 0.0) {
                const double r0 = r_[static_cast<size_t>(ii - 1)], r1 = r_[static_cast<size_t>(ii)];
                const double u = (0.5 * a(pk) - a(ii - 1)) / std::max(1e-300, a(ii) - a(ii - 1));
                const double t = a(pk) / s;
                core = std::max(core, r0 + u * (r1 - r0) + t);
                tcore = std::min(tcore, t);
            }
        }
    }
    if (hi <= 0.0) return map_;  // no profile to adapt to
    RadialMap m;
    m.L = std::max(1e-3, p_.radial_scale_ratio * hi);
    const double c = std::min(0.95, std::max(0.0, p_.radial_cluster_fraction));
    RadialMap::Cluster& cw = m.cl[1];
    cw.ra = std::max(0.0, lo);
    cw.rb = hi;
    const double need_w = (cw.rb - cw.ra) / tmin;
    const double need_c = (core > 0.0) ? core / tcore : 0.0;
    double cc_frac = c * need_c / (need_w + need_c);
    if (core > 0.0) {
        const double floor_frac = std::min(0.5 * c, 16.0 / Nr_);
        cc_frac = std::min(std::max(cc_frac, floor_frac), c - floor_frac);
    }
    const double cw_frac = c - cc_frac;
    cw.beta = cw_frac / (1.0 - c);
    // Soft cluster edges: half the thinnest wall, but at least 3 cluster cells, so that the map
    // derivatives vary slowly on the grid.
    cw.delta = std::max(0.5 * tmin, (cw_frac > 0.0) ? 3.0 * (cw.rb - cw.ra) / (cw_frac * Nr_) : 0.0);
    cw.delta = std::max(cw.delta, 1e-3 * cw.rb);
    if (cc_frac > 0.0) {
        RadialMap::Cluster& cc = m.cl[0];
        cc.ra = -core;
        cc.rb = core;
        cc.beta = cc_frac / (1.0 - c);
        cc.delta = std::max(0.5 * tcore, 3.0 * core / (cc_frac * Nr_));
        cc.delta = std::max(cc.delta, 1e-3 * core);
    }
    return m;
}

bool RotatingSolver::map_drifted(const RadialMap& want) const {
    if (p_.radial_grid != "adaptive") return false;
    if (!map_adapted_) return true;
    const double tol = p_.regrid_tol;
    if (std::abs(want.L - map_.L) > tol * map_.L) return true;
    for (size_t k = 0; k < 2; ++k) {
        const RadialMap::Cluster &a = map_.cl[k], &b = want.cl[k];
        if ((a.beta > 0.0) != (b.beta > 0.0)) return true;
        if (a.beta <= 0.0) continue;
        const double D = a.rb - std::max(0.0, a.ra);  // extent on r >= 0
        if (std::abs(b.beta - a.beta) > tol * a.beta || std::abs(b.ra - a.ra) > tol * D ||
            std::abs(b.rb - a.rb) > tol * D || std::abs(b.delta - a.delta) > tol * D)
            return true;
    }
    return false;
}

RotatingSolver::Vec RotatingSolver::interpolate_state(const Vec& x, const RadialMap& old_map) const {
    // The old nodes are uniform in the old xi: 6-point Lagrange in old xi at old_map.xi(r_new).
    Vec y(x.size());
    y.back() = x.back();
    for (int i = 0; i < Nr_; ++i) {
        int k0 = Nr_ - 1;
        double w[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (i < Nr_ - 1) {
            const double s = (old_map.xi(r_[static_cast<size_t>(i)]) - p_.rbar_min) / hr_;
            k0 = std::min(std::max(static_cast<int>(std::floor(s)) - 2, 0), Nr_ - 6);
            for (int a = 0; a < 6; ++a) {
                w[a] = 1.0;
                for (int b = 0; b < 6; ++b)
                    if (b != a) w[a] *= (s - (k0 + b)) / static_cast<double>(a - b);
            }
        }
        for (int j = 0; j < Nt_; ++j)
            for (int f = 0; f < nf_; ++f) {
                double v = 0.0;
                for (int a = 0; a < 6 && k0 + a < Nr_; ++a) v += w[a] * x[idx(k0 + a, j, f)];
                y[idx(i, j, f)] = v;
            }
    }
    return y;
}

bool RotatingSolver::regrid(const RadialMap& want, const std::vector<Vec*>& states, const std::function<bool()>& resolve) {
    const RadialMap old = map_;
    std::vector<Vec> saved;
    for (Vec* s : states) saved.push_back(*s);
    map_ = want;
    build_grid();
    lu_valid_ = false;
    for (Vec* s : states) *s = interpolate_state(*s, old);
    if (resolve()) {
        map_adapted_ = true;
        ++n_regrids_;
        std::cout << "[info] Regrid: L = " << map_.L;
        const char* names[2] = {"core", "wall"};
        for (int k = 0; k < 2; ++k) {
            const RadialMap::Cluster& cl = map_.cl[static_cast<size_t>(k)];
            if (cl.beta <= 0.0) continue;
            std::cout << "; " << names[k] << " [" << std::max(0.0, cl.ra) << ", " << cl.rb << "] edge " << cl.delta << ", "
                      << 100.0 * map_.fraction(k) << "% of points";
        }
        std::cout << "; r_1 = " << r_[1] << "\n";
        return true;
    }
    std::cout << "[warn] Regrid: re-solve on the new grid failed; keeping the old grid.\n";
    map_ = old;
    build_grid();
    lu_valid_ = false;
    for (size_t k = 0; k < states.size(); ++k) *states[k] = saved[k];
    return false;
}

void RotatingSolver::setup_matter() {
    matter_ = (p_.matter == "proca") ? Matter::Proca : (p_.matter == "dirac") ? Matter::Dirac : Matter::Scalar;
    nf_ = (matter_ == Matter::Scalar) ? 5 : 8;
    theta_full_ = false;  // [0, pi] supported (two poles) but unused: equatorial symmetry removes the z-shift mode
    for (int a = 0; a < NF_MAX; ++a) eq_partner_[a] = a;
    mval_ = (matter_ == Matter::Dirac) ? 0.5 * m_ : static_cast<double>(m_);
    for (int a = 0; a < NF_MAX; ++a) {
        axis_sign_[a] = 1;
        eq_sign_[a] = 1;
        origin_neumann_[a] = false;
    }
    const int sm = (m_ % 2 == 0) ? 1 : -1;  // (-1)^m
    if (matter_ == Matter::Scalar) {
        axis_sign_[F_p] = sm;
        origin_neumann_[F_p] = (m_ == 0);
    } else if (matter_ == Matter::Proca) {
        // A regular vector field with azimuthal number m: V and H1 (A_t, r A_r) behave like
        // (-1)^m under theta -> -theta, H2 and H3 (A_theta, A_varphi / sin) like (-1)^(m+1);
        // equatorial symmetry makes A_theta odd under theta -> pi - theta. Only m = 0 has A_t != 0
        // at the centre.
        axis_sign_[F_V] = axis_sign_[F_H1] = sm;
        axis_sign_[F_H2] = axis_sign_[F_H3] = -sm;
        eq_sign_[F_H2] = -1;
        origin_neumann_[F_V] = (m_ == 0);
    }
    if (matter_ == Matter::Dirac) {
        // m = 1/2: the 1/sin terms of the equations force D2 = D4 = 0 on theta = 0 (D1, D3 even
        // there) and D1 = D3 = 0 on theta = pi (D2, D4 even). The reflection theta -> pi - theta maps
        // the rows into each other with D1 <-> D4, D2 <-> D3 and a common sign (the mass terms carry
        // the signs + - - + in rows D1..D4, so D1 <-> D2 is not a symmetry); dirac_parity picks it.
        axis_sign_[F_D1] = axis_sign_[F_D3] = 1;
        axis_sign_[F_D2] = axis_sign_[F_D4] = -1;
        if (theta_full_) {
            eq_sign_[F_D1] = eq_sign_[F_D3] = -1;  // theta = pi parities
            eq_sign_[F_D2] = eq_sign_[F_D4] = 1;
        } else {
            for (int a = F_D1; a <= F_D4; ++a) eq_sign_[a] = (p_.dirac_parity < 0) ? -1 : 1;
            eq_partner_[F_D1] = F_D4; eq_partner_[F_D4] = F_D1;
            eq_partner_[F_D2] = F_D3; eq_partner_[F_D3] = F_D2;
        }
        if (m_ != 1)
            std::cout << "[warn] Dirac: the pole conditions are derived for m = 1/2 (m_azimuthal = 1) only.\n";
    }
}

void RotatingSolver::seed_analytic() {
    x_.assign(n_unknowns(), 0.0);
    const double A = p_.rot_seed_amp, R = std::max(1e-3, p_.rot_seed_width);
    for (int i = 0; i < Nr_; ++i) {
        for (int j = 0; j < Nt_; ++j) {
            x_[idx(i, j, F_f)] = 1.0;
            x_[idx(i, j, F_l)] = 1.0;
            x_[idx(i, j, F_g)] = 1.0;
            if (i == Nr_ - 1) continue;
            const double r = r_[static_cast<size_t>(i)], th = theta_[static_cast<size_t>(j)];
            const double sn = std::sin(th), cs = std::cos(th);
            if (matter_ == Matter::Scalar) {
                // Flat metric and a torus-shaped scalar field peaked at rho = R, z = 0 (a ball for m = 0).
                const double rho = r * sn, z = r * cs;
                double phi;
                if (m_ == 0) {
                    phi = A * std::exp(-0.5 * r * r / (R * R));
                } else {
                    const double q = rho / R;
                    phi = A * std::pow(q, m_) * std::exp(0.5 * m_ * (1.0 - q * q) - 0.5 * z * z / (R * R));
                }
                x_[idx(i, j, F_p)] = phi;
            } else if (matter_ == Matter::Dirac) {
                // Separable flat-space pattern (m = 1/2): D1 = g cos(theta/2), D2 = g sin(theta/2),
                // D3 = h cos, D4 = h sin, with h = -w g / (mu + k), k = sqrt(mu^2 - w^2) (decaying mode).
                const double gr = A * std::exp(-0.5 * r * r / (R * R));
                const double w0 = p_.omega_target, k = std::sqrt(std::max(0.0, p_.mu * p_.mu - w0 * w0));
                const double hr = -w0 / (p_.mu + k) * gr;
                x_[idx(i, j, F_D1)] = gr * std::cos(0.5 * th);
                x_[idx(i, j, F_D2)] = gr * std::sin(0.5 * th);
                x_[idx(i, j, F_D3)] = hr * std::cos(0.5 * th);
                x_[idx(i, j, F_D4)] = hr * std::sin(0.5 * th);
            } else {
                // Newtonian-limit shape: m = 0, a spherical A_t; m >= 1, the circularly polarised
                // field psi(r) (e_x + i e_y) (times sin^(m-1)): H1 = r psi sin, H2 = r psi cos,
                // H3 = r psi, and V = -psi' sin / omega from the flat-space Lorenz condition.
                const double psi = A * std::exp(-0.5 * r * r / (R * R)), dpsi = -r / (R * R) * psi;
                if (m_ == 0) {
                    // spherical A_t; H1 = -(omega / r) int_0^r V r'^2 dr' (flat Lorenz condition)
                    const double q = r / (std::sqrt(2.0) * R);
                    const double integral = A * R * R * (R * std::sqrt(0.5 * kPi) * std::erf(q) - r * std::exp(-q * q));
                    x_[idx(i, j, F_V)] = psi;
                    x_[idx(i, j, F_H1)] = (r > 0.0) ? -p_.omega_target * integral / r : 0.0;
                } else {
                    const double sm1 = std::pow(sn, m_ - 1);
                    x_[idx(i, j, F_H1)] = r * psi * sn * sm1;
                    x_[idx(i, j, F_H2)] = r * psi * cs * sm1;
                    x_[idx(i, j, F_H3)] = r * psi * sm1;
                    x_[idx(i, j, F_V)] = -dpsi * sn * sm1 / std::max(1e-3, p_.omega_target);
                }
            }
        }
    }
    if (!p_.seed_file.empty()) {
        std::ifstream in(p_.seed_file);
        std::vector<std::vector<double>> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::vector<double> v;
            double d;
            while (ss >> d) v.push_back(d);
            if (v.size() >= 6) rows.push_back(v);
        }
        const bool pattern = !rows.empty() && rows[0].size() == 6 &&
                             ((matter_ == Matter::Proca && m_ >= 1) || matter_ == Matter::Dirac);
        if (rows.size() < 2 || (!pattern && static_cast<int>(rows[0].size()) < 1 + nf_)) {
            std::cout << "[warn] seed_file " << p_.seed_file << ": no usable rows (need r + " << nf_
                      << " columns, or r f l g W psi for Proca m >= 1).\n";
        } else {
            std::cout << "[info] seed from " << p_.seed_file << " (" << rows.size() << " radii"
                      << (pattern ? ", amplitude psi with the m >= 1 Proca pattern" : "") << ")\n";
            auto interp = [&](double r, size_t col, double outside) {
                if (r >= rows.back()[0]) return outside;
                if (r <= rows.front()[0]) return rows.front()[col];
                size_t k = 1;
                while (rows[k][0] < r) ++k;
                const double u = (r - rows[k - 1][0]) / (rows[k][0] - rows[k - 1][0]);
                return (1.0 - u) * rows[k - 1][col] + u * rows[k][col];
            };
            for (int i = 0; i < Nr_; ++i) {
                const double r = r_[static_cast<size_t>(i)];
                const bool inf = (i == Nr_ - 1);
                for (int a = 0; a < (pattern ? 4 : nf_); ++a) {
                    const double outside = (a == F_f || a == F_l || a == F_g) ? 1.0 : 0.0;
                    const double v = inf ? outside : interp(r, static_cast<size_t>(a) + 1, outside);
                    for (int j = 0; j < Nt_; ++j) x_[idx(i, j, a)] = v;
                }
                if (!pattern) continue;
                // psi and psi' (centred difference in r) -> the analytic m >= 1 Proca pattern
                const double psi = inf ? 0.0 : interp(r, 5, 0.0);
                const double dr = 1e-4 * (1.0 + r);
                const double dpsi = inf ? 0.0 : (interp(r + dr, 5, 0.0) - interp(std::max(0.0, r - dr), 5, 0.0)) /
                                                    (r + dr - std::max(0.0, r - dr));
                if (matter_ == Matter::Dirac) {
                    const double w0 = p_.omega_target, k = std::sqrt(std::max(0.0, p_.mu * p_.mu - w0 * w0));
                    for (int j = 0; j < Nt_; ++j) {
                        const double th = theta_[static_cast<size_t>(j)];
                        x_[idx(i, j, F_D1)] = psi * std::cos(0.5 * th);
                        x_[idx(i, j, F_D2)] = psi * std::sin(0.5 * th);
                        x_[idx(i, j, F_D3)] = -w0 / (p_.mu + k) * psi * std::cos(0.5 * th);
                        x_[idx(i, j, F_D4)] = -w0 / (p_.mu + k) * psi * std::sin(0.5 * th);
                    }
                    continue;
                }
                for (int j = 0; j < Nt_; ++j) {
                    const double th = theta_[static_cast<size_t>(j)], sn = std::sin(th), cs = std::cos(th);
                    const double sm1 = std::pow(sn, m_ - 1);
                    x_[idx(i, j, F_H1)] = r * psi * sn * sm1;
                    x_[idx(i, j, F_H2)] = r * psi * cs * sm1;
                    x_[idx(i, j, F_H3)] = r * psi * sm1;
                    x_[idx(i, j, F_V)] = -dpsi * sn * sm1 / std::max(1e-3, p_.omega_target);
                }
            }
        }
    }
    x_.back() = p_.omega_target;
    if (!p_.init_file.empty()) load_init_file();
}

bool RotatingSolver::load_init_file() {
    // bs2d.dat: header "... omega=<w>", rows "i j r theta <nf_ fields>" for the finite radii.
    std::ifstream in(p_.init_file);
    double omega = -1.0;
    std::vector<std::vector<double>> rows;
    std::string line;
    int ni = 0, nj = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t k = line.find(" omega=");
            if (k != std::string::npos) omega = std::stod(line.substr(k + 7));
            if (line.find("# radial map xi(r): L=") == 0) {
                // L=<L> cluster0(fraction=<f> ra=<a> rb=<b> delta=<d>) cluster1(...); beta_k = f_k (1 + sum beta)
                RadialMap mp;
                double fr[2] = {0.0, 0.0};
                bool ok = std::sscanf(line.c_str(), "# radial map xi(r): L=%lf", &mp.L) == 1;
                for (int c = 0; c < 2 && ok; ++c) {
                    const size_t p = line.find("cluster" + std::to_string(c) + "(");
                    RadialMap::Cluster& cl = mp.cl[static_cast<size_t>(c)];
                    ok = p != std::string::npos &&
                         std::sscanf(line.c_str() + p + 9, "fraction=%lf ra=%lf rb=%lf delta=%lf", &fr[c], &cl.ra, &cl.rb, &cl.delta) == 4;
                }
                if (ok && fr[0] + fr[1] < 1.0) {
                    const double S = (fr[0] + fr[1]) / (1.0 - fr[0] - fr[1]);
                    for (int c = 0; c < 2; ++c) mp.cl[static_cast<size_t>(c)].beta = fr[c] * (1.0 + S);
                    init_map_ = mp;
                    init_map_valid_ = true;
                }
            }
            continue;
        }
        std::istringstream ss(line);
        std::vector<double> v;
        double d;
        while (ss >> d) v.push_back(d);
        if (static_cast<int>(v.size()) != 4 + nf_) continue;
        ni = std::max(ni, static_cast<int>(v[0]) + 1);
        nj = std::max(nj, static_cast<int>(v[1]) + 1);
        rows.push_back(v);
    }
    if (omega <= 0.0 || ni < 4 || nj < 4 || static_cast<int>(rows.size()) != ni * nj) {
        std::cout << "[warn] init_file " << p_.init_file << ": not a usable bs2d.dat profile for matter = " << p_.matter
                  << " (need the omega header and " << 4 + nf_ << " columns); using the seed.\n";
        return false;
    }
    std::vector<double> rold(static_cast<size_t>(ni)), told(static_cast<size_t>(nj)), F(rows.size() * static_cast<size_t>(nf_));
    for (const auto& v : rows) {
        const size_t i = static_cast<size_t>(v[0]), j = static_cast<size_t>(v[1]);
        rold[i] = v[2];
        told[j] = v[3];
        for (int a = 0; a < nf_; ++a) F[(i * static_cast<size_t>(nj) + j) * static_cast<size_t>(nf_) + static_cast<size_t>(a)] = v[4 + static_cast<size_t>(a)];
    }
    // 4-point Lagrange weights on the nodes z[k0..k0+3] at z = s
    auto lagrange = [](const std::vector<double>& z, double s, int& k0, double w[4]) {
        const int n = static_cast<int>(z.size());
        int k = static_cast<int>(std::upper_bound(z.begin(), z.end(), s) - z.begin()) - 1;
        k0 = std::min(std::max(k - 1, 0), n - 4);
        for (int a = 0; a < 4; ++a) {
            w[a] = 1.0;
            for (int b = 0; b < 4; ++b)
                if (b != a) w[a] *= (s - z[static_cast<size_t>(k0 + b)]) / (z[static_cast<size_t>(k0 + a)] - z[static_cast<size_t>(k0 + b)]);
        }
    };
    const double rmax = rold.back();
    for (int i = 0; i < Nr_; ++i) {
        const double r = r_[static_cast<size_t>(i)];
        const bool beyond = (i == Nr_ - 1) || r > rmax;
        int ki = 0;
        double wr[4];
        if (!beyond) lagrange(rold, r, ki, wr);
        for (int j = 0; j < Nt_; ++j) {
            int kj = 0;
            double wt[4];
            lagrange(told, theta_[static_cast<size_t>(j)], kj, wt);
            for (int a = 0; a < nf_; ++a) {
                auto at = [&](int ii) {
                    double s = 0.0;
                    for (int b = 0; b < 4; ++b)
                        s += wt[b] * F[(static_cast<size_t>(ii) * static_cast<size_t>(nj) + static_cast<size_t>(kj + b)) * static_cast<size_t>(nf_) + static_cast<size_t>(a)];
                    return s;
                };
                double v;
                if (!beyond) {
                    v = 0.0;
                    for (int b = 0; b < 4; ++b) v += wr[b] * at(ki + b);
                } else if (i == Nr_ - 1) {
                    v = (a == F_f || a == F_l || a == F_g) ? 1.0 : 0.0;
                } else {  // between the old outermost radius and infinity: metric decay, no matter
                    const double last = at(ni - 1);
                    v = (a == F_f || a == F_l || a == F_g) ? 1.0 + (last - 1.0) * rmax / r
                        : (a == F_W) ? last * rmax * rmax / (r * r) : 0.0;
                }
                x_[idx(i, j, a)] = v;
            }
        }
    }
    p_.omega_target = omega;
    x_.back() = omega;
    std::cout << "[info] initial state from " << p_.init_file << " (" << ni << " x " << nj << " nodes, omega = " << omega << ")\n";
    return true;
}

double RotatingSolver::val(const Vec& x, int i, int j, int a) const {
    if (j < 0) {
        const double v = x[idx(i, -j, a)];
        return (axis_sign_[a] < 0) ? -v : v;
    }
    if (j > Nt_ - 1) {
        const double v = x[idx(i, 2 * (Nt_ - 1) - j, eq_partner_[a])];
        return (eq_sign_[a] < 0) ? -v : v;
    }
    return x[idx(i, j, a)];
}

double RotatingSolver::amp(const Vec& x, int i, int j) const {
    if (matter_ == Matter::Scalar) return std::abs(x[idx(i, j, F_p)]);
    double s2 = 0.0;
    for (int a = 4; a < nf_; ++a) s2 += x[idx(i, j, a)] * x[idx(i, j, a)];
    return std::sqrt(s2);
}

void RotatingSolver::deriv_r(const Vec& x, int i, int j, int a, double& d1, double& d2, double& a1, double& a2) const {
    // Stencils in xi (as the 1D solver: 5-point, 3-point next to the boundaries, one-sided at
    // r = 0); physical derivatives via d/dr = xi' d/dxi, d^2/dr^2 = xi'^2 d^2/dxi^2 + xi'' d/dxi.
    const double h = hr_;
    if (i >= 2 && i <= Nr_ - 3) {
        const double u0 = val(x, i - 2, j, a), u1 = val(x, i - 1, j, a), u2 = val(x, i, j, a),
                     u3 = val(x, i + 1, j, a), u4 = val(x, i + 2, j, a);
        d1 = (u0 / 12.0 - 2.0 * u1 / 3.0 + 2.0 * u3 / 3.0 - u4 / 12.0) / h;
        d2 = (-u0 / 12.0 + 4.0 * u1 / 3.0 - 2.5 * u2 + 4.0 * u3 / 3.0 - u4 / 12.0) / (h * h);
        a1 = (std::abs(u0) / 12.0 + 2.0 * std::abs(u1) / 3.0 + 2.0 * std::abs(u3) / 3.0 + std::abs(u4) / 12.0) / h;
        a2 = (std::abs(u0) / 12.0 + 4.0 * std::abs(u1) / 3.0 + 2.5 * std::abs(u2) + 4.0 * std::abs(u3) / 3.0
              + std::abs(u4) / 12.0) / (h * h);
    } else if (i >= 1) {
        const double um = val(x, i - 1, j, a), u = val(x, i, j, a), up = val(x, i + 1, j, a);
        d1 = (up - um) / (2.0 * h);
        d2 = (up - 2.0 * u + um) / (h * h);
        a1 = (std::abs(up) + std::abs(um)) / (2.0 * h);
        a2 = (std::abs(up) + 2.0 * std::abs(u) + std::abs(um)) / (h * h);
    } else {
        const double u0 = val(x, 0, j, a), u1 = val(x, 1, j, a), u2 = val(x, 2, j, a);
        d1 = (-3.0 * u0 + 4.0 * u1 - u2) / (2.0 * h);
        d2 = (u0 - 2.0 * u1 + u2) / (h * h);
        a1 = (3.0 * std::abs(u0) + 4.0 * std::abs(u1) + std::abs(u2)) / (2.0 * h);
        a2 = (std::abs(u0) + 2.0 * std::abs(u1) + std::abs(u2)) / (h * h);
    }
    const double xr = xi_r_[static_cast<size_t>(i)], xrr = xi_rr_[static_cast<size_t>(i)];
    const double d1b = d1, a1b = a1;
    d1 = xr * d1b;
    d2 = xr * xr * d2 + xrr * d1b;
    a1 = xr * a1b;
    a2 = xr * xr * a2 + std::abs(xrr) * a1b;
}

void RotatingSolver::deriv_t(const Vec& x, int i, int j, int a, double& d1, double& d2, double& a1, double& a2) const {
    const double h = ht_;
    const double u0 = val(x, i, j - 2, a), u1 = val(x, i, j - 1, a), u2 = val(x, i, j, a),
                 u3 = val(x, i, j + 1, a), u4 = val(x, i, j + 2, a);
    d1 = (u0 / 12.0 - 2.0 * u1 / 3.0 + 2.0 * u3 / 3.0 - u4 / 12.0) / h;
    d2 = (-u0 / 12.0 + 4.0 * u1 / 3.0 - 2.5 * u2 + 4.0 * u3 / 3.0 - u4 / 12.0) / (h * h);
    a1 = (std::abs(u0) / 12.0 + 2.0 * std::abs(u1) / 3.0 + 2.0 * std::abs(u3) / 3.0 + std::abs(u4) / 12.0) / h;
    a2 = (std::abs(u0) / 12.0 + 4.0 * std::abs(u1) / 3.0 + 2.5 * std::abs(u2) + 4.0 * std::abs(u3) / 3.0
          + std::abs(u4) / 12.0) / (h * h);
}

namespace {
// first-derivative weights (times h) on offsets -1..3 (forward-biased, 4th order); reversed and
// negated for backward on -3..1
constexpr double kFwd4[5] = {-3.0 / 12.0, -10.0 / 12.0, 18.0 / 12.0, -6.0 / 12.0, 1.0 / 12.0};
}

void RotatingSolver::deriv_r_biased(const Vec& x, int i, int j, int a, int dir, double& d1, double& a1) const {
    const double h = hr_;
    d1 = a1 = 0.0;
    auto acc = [&](int off, double wgt) {
        const double u = val(x, i + off, j, a);
        d1 += wgt * u;
        a1 += std::abs(wgt * u);
    };
    if (dir > 0) {
        if (i >= 1 && i + 3 <= Nr_ - 1) for (int k = 0; k < 5; ++k) acc(k - 1, kFwd4[k]);
        else if (i + 1 <= Nr_ - 1 && i >= 1) { acc(1, 0.5); acc(-1, -0.5); }  // near the outer boundary
        else { acc(0, -1.5); acc(1, 2.0); acc(2, -0.5); }                     // i = 0: one-sided
    } else {
        if (i >= 3 && i + 1 <= Nr_ - 1) for (int k = 0; k < 5; ++k) acc(1 - k, -kFwd4[k]);
        else if (i == 2) { acc(-2, 1.0 / 6.0); acc(-1, -1.0); acc(0, 0.5); acc(1, 1.0 / 3.0); }  // 3rd order on -2..1
        else if (i == 1) { acc(-1, -1.0); acc(0, 1.0); }                                         // 1st order
        else { acc(0, -1.5); acc(1, 2.0); acc(2, -0.5); }                                        // i = 0
    }
    const double xr = xi_r_[static_cast<size_t>(i)] / h;
    d1 *= xr;
    a1 *= xr;
}

void RotatingSolver::deriv_t_biased(const Vec& x, int i, int j, int a, int dir, double& d1, double& a1) const {
    d1 = a1 = 0.0;
    for (int k = 0; k < 5; ++k) {
        const int off = (dir > 0) ? k - 1 : 1 - k;
        const double wgt = (dir > 0) ? kFwd4[k] : -kFwd4[k];
        const double u = val(x, i, j + off, a);
        d1 += wgt * u;
        a1 += std::abs(wgt * u);
    }
    d1 /= ht_;
    a1 /= ht_;
}

void RotatingSolver::deriv_rt(const Vec& x, int i, int j, int a, double& d, double& ad) const {
    // d^2/(dr dtheta) = xi' d^2/(dxi dtheta): tensor product of the first-derivative stencils
    // (5-point in theta; in xi 5-point, 3-point next to the radial boundaries, one-sided at r = 0).
    static const double c5[5] = {1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0};
    static const double c3[5] = {0.0, -0.5, 0.0, 0.5, 0.0};
    static const double c1[5] = {0.0, 0.0, -1.5, 2.0, -0.5};  // offsets 0, 1, 2 at r = 0
    const double* cr = (i >= 2 && i <= Nr_ - 3) ? c5 : (i >= 1 ? c3 : c1);
    d = 0.0;
    ad = 0.0;
    for (int p = 0; p < 5; ++p) {
        if (cr[p] == 0.0) continue;
        for (int q = 0; q < 5; ++q) {
            if (c5[q] == 0.0) continue;
            const double u = val(x, i + p - 2, j + q - 2, a);
            d += cr[p] * c5[q] * u;
            ad += std::abs(cr[p] * c5[q] * u);
        }
    }
    const double xr = xi_r_[static_cast<size_t>(i)] / (hr_ * ht_);
    d *= xr;
    ad *= xr;
}

void RotatingSolver::node_rows(const Vec& x, int i, int j, double* Rn, double* Sn) const {
    // The nf_ residual rows (and, if Sn, their |term| scales) of node (i, j).
    const double omega = x.back();
    const double kappa = kappa_;
    const double m = mval_;

    // One-sided derivatives used by the boundary rows: rbar (2nd order, as in 1D) and
    // theta on the axis (4th order).
    auto row_d1r0 = [&](int j, int a, double& sc) {
        const double u0 = x[idx(0, j, a)], u1 = x[idx(1, j, a)], u2 = x[idx(2, j, a)];
        sc = (3.0 * std::abs(u0) + 4.0 * std::abs(u1) + std::abs(u2)) / (2.0 * hr_);
        return (-3.0 * u0 + 4.0 * u1 - u2) / (2.0 * hr_);
    };
    auto row_d1t0 = [&](int i, int a, double& sc) {
        double u[5];
        for (int k = 0; k < 5; ++k) u[k] = x[idx(i, k, a)];
        sc = (25.0 * std::abs(u[0]) + 48.0 * std::abs(u[1]) + 36.0 * std::abs(u[2]) + 16.0 * std::abs(u[3])
              + 3.0 * std::abs(u[4])) / (12.0 * ht_);
        return (-25.0 * u[0] + 48.0 * u[1] - 36.0 * u[2] + 16.0 * u[3] - 3.0 * u[4]) / (12.0 * ht_);
    };
    auto row_d1tN = [&](int i, int a, double& sc) {  // backward, at theta = pi
        double u[5];
        for (int k = 0; k < 5; ++k) u[k] = x[idx(i, Nt_ - 1 - k, a)];
        sc = (25.0 * std::abs(u[0]) + 48.0 * std::abs(u[1]) + 36.0 * std::abs(u[2]) + 16.0 * std::abs(u[3])
              + 3.0 * std::abs(u[4])) / (12.0 * ht_);
        return (25.0 * u[0] - 48.0 * u[1] + 36.0 * u[2] - 16.0 * u[3] + 3.0 * u[4]) / (12.0 * ht_);
    };
    const size_t base = idx(i, j, 0);
    auto set = [&](size_t row, double rv, double sv) {
        Rn[row - base] = rv;
        if (Sn) Sn[row - base] = sv;
    };
    auto set_zero = [&](int a) { const double v = x[idx(i, j, a)]; set(idx(i, j, a), v, std::abs(v)); };
    [[maybe_unused]] const bool want_scale = (Sn != nullptr);

    const double fv = x[idx(i, j, F_f)], lv = x[idx(i, j, F_l)], gv = x[idx(i, j, F_g)], Wv = x[idx(i, j, F_W)];
    double sc = 0.0;
    if (i == Nr_ - 1) {
        // infinity: f = l = g = 1, W = 0, matter fields 0
        set(idx(i, j, F_f), fv - 1.0, std::abs(fv) + 1.0);
        set(idx(i, j, F_l), lv - 1.0, std::abs(lv) + 1.0);
        set(idx(i, j, F_g), gv - 1.0, std::abs(gv) + 1.0);
        set(idx(i, j, F_W), Wv, std::abs(Wv));
        for (int a = 4; a < nf_; ++a) set_zero(a);
        return;
    }
    bool metric_done = false;
    if (i == 0) {
        // origin: d_r f = d_r l = 0, g = 1, W = 0; matter fields 0 (d_r = 0 for central m = 0 fields)
        double rv = row_d1r0(j, F_f, sc); set(idx(i, j, F_f), rv, sc);
        rv = row_d1r0(j, F_l, sc); set(idx(i, j, F_l), rv, sc);
        set(idx(i, j, F_g), gv - 1.0, std::abs(gv) + 1.0);
        set(idx(i, j, F_W), Wv, std::abs(Wv));
        if (matter_ == Matter::Dirac && p_.dirac_origin_zero == 0 && j > 0 && (j < Nt_ - 1 || !theta_full_)) {
            // Dirac: the regular spinor is nonzero at r = 0 (D1 : D2 = D3 : D4 = cos : sin of theta/2,
            // radial exponent 0). The r-weighted equations at r = 0 lose r d_r and fix only that
            // ratio, not the amplitude (on the grid they pin D = 0); instead: smoothness along each
            // ray, quadratic extrapolation from i = 1, 2, 3.
            for (int a = 4; a < nf_; ++a) {
                const double u0 = x[idx(0, j, a)], u1 = x[idx(1, j, a)], u2 = x[idx(2, j, a)], u3 = x[idx(3, j, a)];
                set(idx(i, j, a), u0 - (3.0 * u1 - 3.0 * u2 + u3),
                    std::abs(u0) + 3.0 * std::abs(u1) + 3.0 * std::abs(u2) + std::abs(u3));
            }
            return;
        } else {
            for (int a = 4; a < nf_; ++a) {
                if (matter_ == Matter::Dirac && p_.dirac_origin_zero != 0) { set_zero(a); continue; }
                if (matter_ == Matter::Dirac) {  // origin on a pole: the pole conditions
                    const int sgn = (j == 0) ? axis_sign_[a] : eq_sign_[a];
                    if (sgn > 0) { rv = (j == 0) ? row_d1t0(i, a, sc) : row_d1tN(i, a, sc); set(idx(i, j, a), rv, sc); }
                    else set_zero(a);
                    continue;
                }
                if (origin_neumann_[a]) { rv = row_d1r0(j, a, sc); set(idx(i, j, a), rv, sc); }
                else set_zero(a);
            }
            return;
        }
    }
    if (j == 0) {
        // axis: d_theta f = d_theta l = d_theta W = 0, g = 1; matter: odd fields 0, even d_theta = 0
        double rv = row_d1t0(i, F_f, sc); set(idx(i, j, F_f), rv, sc);
        rv = row_d1t0(i, F_l, sc); set(idx(i, j, F_l), rv, sc);
        set(idx(i, j, F_g), gv - 1.0, std::abs(gv) + 1.0);
        rv = row_d1t0(i, F_W, sc); set(idx(i, j, F_W), rv, sc);
        for (int a = 4; a < nf_; ++a) {
            if (axis_sign_[a] > 0) { rv = row_d1t0(i, a, sc); set(idx(i, j, a), rv, sc); }
            else set_zero(a);
        }
        return;
    }
    if (theta_full_ && j == Nt_ - 1) {
        // second pole theta = pi: as the axis, with the theta = pi parities (eq_sign_)
        double rv = row_d1tN(i, F_f, sc); set(idx(i, j, F_f), rv, sc);
        rv = row_d1tN(i, F_l, sc); set(idx(i, j, F_l), rv, sc);
        set(idx(i, j, F_g), gv - 1.0, std::abs(gv) + 1.0);
        rv = row_d1tN(i, F_W, sc); set(idx(i, j, F_W), rv, sc);
        for (int a = 4; a < nf_; ++a) {
            if (eq_sign_[a] > 0) { rv = row_d1tN(i, a, sc); set(idx(i, j, a), rv, sc); }
            else set_zero(a);
        }
        return;
    }

    // Interior and equator (mirrored ghosts there): field equations.
    [[maybe_unused]] const double fs = fv, ls = lv, gs = gv, Ws = Wv;
    [[maybe_unused]] const double r = r_[static_cast<size_t>(i)];
    [[maybe_unused]] const double sin_t = std::sin(theta_[static_cast<size_t>(j)]);
    [[maybe_unused]] const double cos_t = std::cos(theta_[static_cast<size_t>(j)]);
    [[maybe_unused]] double f_r, f_rr, fa_r, fa_rr, f_t, f_tt, fa_t, fa_tt;
    [[maybe_unused]] double l_r, l_rr, la_r, la_rr, l_t, l_tt, la_t, la_tt;
    [[maybe_unused]] double g_r, g_rr, ga_r, ga_rr, g_t, g_tt, ga_t, ga_tt;
    [[maybe_unused]] double W_r, W_rr, Wa_r, Wa_rr, W_t, W_tt, Wa_t, Wa_tt;
    deriv_r(x, i, j, F_f, f_r, f_rr, fa_r, fa_rr); deriv_t(x, i, j, F_f, f_t, f_tt, fa_t, fa_tt);
    deriv_r(x, i, j, F_l, l_r, l_rr, la_r, la_rr); deriv_t(x, i, j, F_l, l_t, l_tt, la_t, la_tt);
    deriv_r(x, i, j, F_g, g_r, g_rr, ga_r, ga_rr); deriv_t(x, i, j, F_g, g_t, g_tt, ga_t, ga_tt);
    deriv_r(x, i, j, F_W, W_r, W_rr, Wa_r, Wa_rr); deriv_t(x, i, j, F_W, W_t, W_tt, Wa_t, Wa_tt);
    double res_f, res_l, res_g, res_W, scale_f = 0, scale_l = 0, scale_g = 0, scale_W = 0;
    if (matter_ == Matter::Scalar) {
        [[maybe_unused]] const double ps = x[idx(i, j, F_p)];
        [[maybe_unused]] const double U = potential_U(p_, ps);
        [[maybe_unused]] const double dU = potential_dU(p_, ps);
        [[maybe_unused]] const double dUa = potential_dU_abs_terms(p_, ps);
        [[maybe_unused]] double p_r, p_rr, pa_r, pa_rr, p_t, p_tt, pa_t, pa_tt;
        deriv_r(x, i, j, F_p, p_r, p_rr, pa_r, pa_rr); deriv_t(x, i, j, F_p, p_t, p_tt, pa_t, pa_tt);
        double res_p, scale_p = 0;
#include "rotating_equations.inc"
        set(idx(i, j, F_p), res_p, scale_p);
    } else if (matter_ == Matter::Proca) {
        [[maybe_unused]] const double mu = p_.mu;
        [[maybe_unused]] const double Vs = x[idx(i, j, F_V)], H1s = x[idx(i, j, F_H1)], H2s = x[idx(i, j, F_H2)],
                                      H3s = x[idx(i, j, F_H3)];
        [[maybe_unused]] double V_r, V_rr, Va_r, Va_rr, V_t, V_tt, Va_t, Va_tt;
        [[maybe_unused]] double H1_r, H1_rr, H1a_r, H1a_rr, H1_t, H1_tt, H1a_t, H1a_tt;
        [[maybe_unused]] double H2_r, H2_rr, H2a_r, H2a_rr, H2_t, H2_tt, H2a_t, H2a_tt;
        [[maybe_unused]] double H3_r, H3_rr, H3a_r, H3a_rr, H3_t, H3_tt, H3a_t, H3a_tt;
        [[maybe_unused]] double H1_rt, H1a_rt, H2_rt, H2a_rt;
        deriv_r(x, i, j, F_V, V_r, V_rr, Va_r, Va_rr); deriv_t(x, i, j, F_V, V_t, V_tt, Va_t, Va_tt);
        deriv_r(x, i, j, F_H1, H1_r, H1_rr, H1a_r, H1a_rr); deriv_t(x, i, j, F_H1, H1_t, H1_tt, H1a_t, H1a_tt);
        deriv_r(x, i, j, F_H2, H2_r, H2_rr, H2a_r, H2a_rr); deriv_t(x, i, j, F_H2, H2_t, H2_tt, H2a_t, H2a_tt);
        deriv_r(x, i, j, F_H3, H3_r, H3_rr, H3a_r, H3a_rr); deriv_t(x, i, j, F_H3, H3_t, H3_tt, H3a_t, H3a_tt);
        deriv_rt(x, i, j, F_H1, H1_rt, H1a_rt);
        deriv_rt(x, i, j, F_H2, H2_rt, H2a_rt);
        double res_V, res_H1, res_H2, res_H3, scale_V = 0, scale_H1 = 0, scale_H2 = 0, scale_H3 = 0;
#include "proca_equations.inc"
        set(idx(i, j, F_V), res_V, scale_V);
        set(idx(i, j, F_H1), res_H1, scale_H1);
        set(idx(i, j, F_H2), res_H2, scale_H2);
        set(idx(i, j, F_H3), res_H3, scale_H3);
    } else {
        [[maybe_unused]] const double mu = p_.mu;
        [[maybe_unused]] const double D1s = x[idx(i, j, F_D1)], D2s = x[idx(i, j, F_D2)], D3s = x[idx(i, j, F_D3)],
                                      D4s = x[idx(i, j, F_D4)];
        [[maybe_unused]] double D1_r, D1_rr, D1a_r, D1a_rr, D1_t, D1_tt, D1a_t, D1a_tt;
        [[maybe_unused]] double D2_r, D2_rr, D2a_r, D2a_rr, D2_t, D2_tt, D2a_t, D2a_tt;
        [[maybe_unused]] double D3_r, D3_rr, D3a_r, D3a_rr, D3_t, D3_tt, D3a_t, D3a_tt;
        [[maybe_unused]] double D4_r, D4_rr, D4a_r, D4a_rr, D4_t, D4_tt, D4a_t, D4a_tt;
        // Each derivative symbol enters one row only (D1-row: D1_r, D2_t; D2-row: D2_r, D1_t; same for
        // D3, D4): forward d_r with backward d_theta in one row of a pair, the reverse in the other --
        // the pair then behaves like a staggered grid and has no checkerboard null modes.
        D1_rr = D2_rr = D3_rr = D4_rr = D1_tt = D2_tt = D3_tt = D4_tt = 0.0;
        D1a_rr = D2a_rr = D3a_rr = D4a_rr = D1a_tt = D2a_tt = D3a_tt = D4a_tt = 0.0;
        deriv_r_biased(x, i, j, F_D1, +1, D1_r, D1a_r); deriv_t_biased(x, i, j, F_D2, -1, D2_t, D2a_t);
        deriv_r_biased(x, i, j, F_D2, -1, D2_r, D2a_r); deriv_t_biased(x, i, j, F_D1, +1, D1_t, D1a_t);
        deriv_r_biased(x, i, j, F_D3, +1, D3_r, D3a_r); deriv_t_biased(x, i, j, F_D4, -1, D4_t, D4a_t);
        deriv_r_biased(x, i, j, F_D4, -1, D4_r, D4a_r); deriv_t_biased(x, i, j, F_D3, +1, D3_t, D3a_t);
        double res_D1, res_D2, res_D3, res_D4, scale_D1 = 0, scale_D2 = 0, scale_D3 = 0, scale_D4 = 0;
#include "dirac_equations.inc"
        if (p_.dirac_dissipation > 0.0) {
            // fourth differences of each field in its own row (see bs_params.h dirac_dissipation)
            const double sig = p_.dirac_dissipation;
            double* res[4] = {&res_D1, &res_D2, &res_D3, &res_D4};
            double* scl[4] = {&scale_D1, &scale_D2, &scale_D3, &scale_D4};
            for (int q = 0; q < 4; ++q) {
                const int a = F_D1 + q;
                double d4 = 0.0, a4 = 0.0;
                const double c4[5] = {1.0, -4.0, 6.0, -4.0, 1.0};
                if (i >= 2 && i <= Nr_ - 3) {
                    const double wr = r * xi_r_[static_cast<size_t>(i)] / hr_;
                    for (int k = 0; k < 5; ++k) {
                        const double u = val(x, i + k - 2, j, a);
                        d4 += wr * c4[k] * u;
                        a4 += wr * std::abs(c4[k] * u);
                    }
                }
                for (int k = 0; k < 5; ++k) {
                    const double u = val(x, i, j + k - 2, a);
                    d4 += c4[k] * u / ht_;
                    a4 += std::abs(c4[k] * u) / ht_;
                }
                *res[q] -= sig * d4;
                if (want_scale) *scl[q] += sig * a4;
            }
        }
        set(idx(i, j, F_D1), res_D1, scale_D1);
        set(idx(i, j, F_D2), res_D2, scale_D2);
        set(idx(i, j, F_D3), res_D3, scale_D3);
        set(idx(i, j, F_D4), res_D4, scale_D4);
    }
    if (metric_done) return;
    set(idx(i, j, F_f), res_f, scale_f);
    set(idx(i, j, F_l), res_l, scale_l);
    set(idx(i, j, F_g), res_g, scale_g);
    set(idx(i, j, F_W), res_W, scale_W);
    if (!theta_full_ && j == Nt_ - 1)  // equator: fields odd under theta -> pi - theta vanish there
        for (int a = 4; a < nf_; ++a) {
            if (eq_partner_[a] != a) {
                if (eq_partner_[a] < a) {  // second of a pair: equal to its partner (sign) on the equator
                    const double v = x[idx(i, j, a)], w = x[idx(i, j, eq_partner_[a])] * eq_sign_[a];
                    set(idx(i, j, a), v - w, std::abs(v) + std::abs(w));
                }
                continue;
            }
            if (eq_sign_[a] < 0) set_zero(a);
        }
}

void RotatingSolver::residual(const Vec& x, Vec& R, Vec* S) const {
    const size_t n = n_unknowns();
    R.assign(n, 0.0);
    if (S) S->assign(n, 0.0);
    const double omega = x.back();
    auto set = [&](size_t row, double rv, double sv) {
        R[row] = rv;
        if (S) (*S)[row] = sv;
    };
    // Rows of different nodes are independent: parallel over r (OpenMP) on grids large enough
    // to amortize the threading overhead. Inside a parallel region this runs serially.
#pragma omp parallel for schedule(static) if (Nr_ * Nt_ >= 4000)
    for (int i = 0; i < Nr_; ++i)
        for (int j = 0; j < Nt_; ++j)
            node_rows(x, i, j, &R[idx(i, j, 0)], S ? &(*S)[idx(i, j, 0)] : nullptr);

    // Constraint row closing the system for the scalar omega.
    const size_t last = n - 1;
    if (constraint_ == Constraint::Omega) {
        set(last, omega - omega_target_, std::abs(omega) + std::abs(omega_target_));
    } else if (constraint_ == Constraint::Arclength) {
        double sv = 0.0, sa = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sv += arc_t_[k] * (x[k] - arc_pred_[k]);
            sa += std::abs(arc_t_[k]) * (std::abs(x[k]) + std::abs(arc_pred_[k]));
        }
        set(last, sv, sa);
    } else {
        const double Q = charge(x);
        set(last, Q - charge_target_, std::abs(Q) + std::abs(charge_target_));
    }
}

void RotatingSolver::node_densities(const Vec& x, int i, int j, double& komar, double& charge, double& angmom,
                                    double* lorenz_out, double* lorenz_scale_out) const {
    [[maybe_unused]] const double omega = x.back(), m = mval_;
    [[maybe_unused]] const double fs = x[idx(i, j, F_f)], ls = x[idx(i, j, F_l)], gs = x[idx(i, j, F_g)],
                                  Ws = x[idx(i, j, F_W)];
    [[maybe_unused]] const double r = r_[static_cast<size_t>(i)];
    [[maybe_unused]] const double sin_t = std::sin(theta_[static_cast<size_t>(j)]);
    [[maybe_unused]] const double cos_t = std::cos(theta_[static_cast<size_t>(j)]);
    double dens_komar = 0, dens_charge = 0, dens_angmom = 0;
    if (matter_ == Matter::Scalar) {
        [[maybe_unused]] const double ps = x[idx(i, j, F_p)];
        [[maybe_unused]] const double U = potential_U(p_, ps);
#include "rotating_densities.inc"
    } else if (matter_ == Matter::Dirac) {
        if (i == 0) {  // the densities carry r^2
            komar = charge = angmom = 0.0;
            if (lorenz_out) *lorenz_out = 0.0;
            if (lorenz_scale_out) *lorenz_scale_out = 0.0;
            return;
        }
        [[maybe_unused]] const double mu = p_.mu;
        [[maybe_unused]] const double D1s = x[idx(i, j, F_D1)], D2s = x[idx(i, j, F_D2)], D3s = x[idx(i, j, F_D3)],
                                      D4s = x[idx(i, j, F_D4)];
        [[maybe_unused]] double d2, a1, a2, f_r, f_t, l_r, l_t, g_r, g_t, W_r, W_t;
        [[maybe_unused]] double D1_r, D1_t, D2_r, D2_t, D3_r, D3_t, D4_r, D4_t;
        deriv_r(x, i, j, F_f, f_r, d2, a1, a2); deriv_t(x, i, j, F_f, f_t, d2, a1, a2);
        deriv_r(x, i, j, F_l, l_r, d2, a1, a2); deriv_t(x, i, j, F_l, l_t, d2, a1, a2);
        deriv_r(x, i, j, F_g, g_r, d2, a1, a2); deriv_t(x, i, j, F_g, g_t, d2, a1, a2);
        deriv_r(x, i, j, F_W, W_r, d2, a1, a2); deriv_t(x, i, j, F_W, W_t, d2, a1, a2);
        deriv_r(x, i, j, F_D1, D1_r, d2, a1, a2); deriv_t(x, i, j, F_D1, D1_t, d2, a1, a2);
        deriv_r(x, i, j, F_D2, D2_r, d2, a1, a2); deriv_t(x, i, j, F_D2, D2_t, d2, a1, a2);
        deriv_r(x, i, j, F_D3, D3_r, d2, a1, a2); deriv_t(x, i, j, F_D3, D3_t, d2, a1, a2);
        deriv_r(x, i, j, F_D4, D4_r, d2, a1, a2); deriv_t(x, i, j, F_D4, D4_t, d2, a1, a2);
#include "dirac_densities.inc"
        if (lorenz_out) *lorenz_out = 0.0;
        if (lorenz_scale_out) *lorenz_scale_out = 0.0;
    } else {
        [[maybe_unused]] const double mu = p_.mu;
        [[maybe_unused]] const double Vs = x[idx(i, j, F_V)], H1s = x[idx(i, j, F_H1)], H2s = x[idx(i, j, F_H2)],
                                      H3s = x[idx(i, j, F_H3)];
        double d2, a1, a2, l_r, l_t, V_r, V_t, H1_r, H1_t, H2_r, H2_t, H3_r, H3_t;
        deriv_r(x, i, j, F_l, l_r, d2, a1, a2); deriv_t(x, i, j, F_l, l_t, d2, a1, a2);
        deriv_r(x, i, j, F_V, V_r, d2, a1, a2); deriv_t(x, i, j, F_V, V_t, d2, a1, a2);
        deriv_r(x, i, j, F_H1, H1_r, d2, a1, a2); deriv_t(x, i, j, F_H1, H1_t, d2, a1, a2);
        deriv_r(x, i, j, F_H2, H2_r, d2, a1, a2); deriv_t(x, i, j, F_H2, H2_t, d2, a1, a2);
        deriv_r(x, i, j, F_H3, H3_r, d2, a1, a2); deriv_t(x, i, j, F_H3, H3_t, d2, a1, a2);
        double lorenz = 0, lorenz_scale = 0;
        if (i == 0) {  // r = 0: the densities carry r^2 and vanish; the Lorenz terms are singular
            komar = charge = angmom = 0.0;
            if (lorenz_out) *lorenz_out = 0.0;
            if (lorenz_scale_out) *lorenz_scale_out = 0.0;
            return;
        }
#include "proca_densities.inc"
        if (lorenz_out) *lorenz_out = lorenz;
        if (lorenz_scale_out) *lorenz_scale_out = lorenz_scale;
    }
    komar = dens_komar;
    charge = dens_charge;
    angmom = dens_angmom;
}

double RotatingSolver::charge(const Vec& x) const {
    double Q = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : Q)
    for (int i = 0; i < Nr_ - 1; ++i) {
        for (int j = 0; j < Nt_; ++j) {
            double k, q, a;
            if (j == 0 && matter_ != Matter::Scalar) {  // axis: extrapolate, as in observables()
                double qq[5];
                for (int jj = 1; jj <= 4; ++jj) node_densities(x, i, jj, k, qq[jj], a);
                q = 4.0 * qq[1] - 6.0 * qq[2] + 4.0 * qq[3] - qq[4];
            } else if (theta_full_ && j == Nt_ - 1) {
                double qq[5];
                for (int jj = 1; jj <= 4; ++jj) node_densities(x, i, Nt_ - 1 - jj, k, qq[jj], a);
                q = 4.0 * qq[1] - 6.0 * qq[2] + 4.0 * qq[3] - qq[4];
            } else {
                node_densities(x, i, j, k, q, a);
            }
            Q += wq_[static_cast<size_t>(i) * Nt_ + j] * q;
        }
    }
    return Q;
}

void RotatingSolver::charge_gradient(const Vec& x, Vec& grad) const {
    // dens_charge = base phi^2 (W m + omega r), base = 2 g l^{3/2} r sin / f^2 (node-local).
    grad.assign(n_unknowns(), 0.0);
    const double omega = x.back(), m = static_cast<double>(m_);
    for (int i = 0; i + 1 < Nr_; ++i) {
        for (int j = 0; j < Nt_; ++j) {
            const double w = wq_[static_cast<size_t>(i) * Nt_ + j];
            const double f = x[idx(i, j, F_f)], l = x[idx(i, j, F_l)], g = x[idx(i, j, F_g)],
                         W = x[idx(i, j, F_W)], p = x[idx(i, j, F_p)];
            const double r = r_[static_cast<size_t>(i)], sn = std::sin(theta_[static_cast<size_t>(j)]);
            const double base = 2.0 * g * std::pow(l, 1.5) * r * sn / (f * f);
            const double D = base * p * p * (W * m + omega * r);
            grad[idx(i, j, F_f)] += w * (-2.0 * D / f);
            grad[idx(i, j, F_l)] += w * (1.5 * D / l);
            grad[idx(i, j, F_g)] += w * (D / g);
            grad[idx(i, j, F_W)] += w * base * p * p * m;
            grad[idx(i, j, F_p)] += w * 2.0 * base * p * (W * m + omega * r);
            grad.back() += w * base * p * p * r;
        }
    }
}

RotatingSolver::Observables RotatingSolver::observables(const Vec& x) const {
    Observables o;
    [[maybe_unused]] const double omega = x.back(), m = static_cast<double>(m_);
    o.omega = omega;
    std::vector<double> shell_M(static_cast<size_t>(Nr_), 0.0), shell_Q(static_cast<size_t>(Nr_), 0.0);
    double lor_max = 0.0, lor_scale_max = 0.0;
    for (int i = 0; i + 1 < Nr_; ++i) {
        for (int j = 0; j < Nt_; ++j) {
            double dens_komar, dens_charge, dens_angmom, lor = 0.0, lor_scale = 0.0;
            node_densities(x, i, j, dens_komar, dens_charge, dens_angmom, &lor, &lor_scale);
            if (j == 0 && matter_ != Matter::Scalar) {
                // On the axis sqrt(-g) ~ sin(theta) makes the Proca densities 0/0: extrapolate
                // (cubic) from theta nodes 1..4; the Lorenz condition is not evaluated there.
                double k[5], q[5], a[5];
                for (int jj = 1; jj <= 4; ++jj) node_densities(x, i, jj, k[jj], q[jj], a[jj]);
                dens_komar = 4.0 * k[1] - 6.0 * k[2] + 4.0 * k[3] - k[4];
                dens_charge = 4.0 * q[1] - 6.0 * q[2] + 4.0 * q[3] - q[4];
                dens_angmom = 4.0 * a[1] - 6.0 * a[2] + 4.0 * a[3] - a[4];
                lor = lor_scale = 0.0;
            }
            if (theta_full_ && j == Nt_ - 1) {  // second pole: likewise from nodes Nt-2 .. Nt-5
                double k[5], q[5], a[5];
                for (int jj = 1; jj <= 4; ++jj) node_densities(x, i, Nt_ - 1 - jj, k[jj], q[jj], a[jj]);
                dens_komar = 4.0 * k[1] - 6.0 * k[2] + 4.0 * k[3] - k[4];
                dens_charge = 4.0 * q[1] - 6.0 * q[2] + 4.0 * q[3] - q[4];
                dens_angmom = 4.0 * a[1] - 6.0 * a[2] + 4.0 * a[3] - a[4];
                lor = lor_scale = 0.0;
            }
            lor_max = std::max(lor_max, std::abs(lor));
            lor_scale_max = std::max(lor_scale_max, lor_scale);
            const double w = wq_[static_cast<size_t>(i) * Nt_ + j];
            o.mass_komar += w * dens_komar;
            o.charge += w * dens_charge;
            o.angmom += w * dens_angmom;
            o.phi_max = std::max(o.phi_max, amp(x, i, j));
            // per-radius (theta-integrated) densities for the R_99 radii below
            const double wth = wq_th_[static_cast<size_t>(j)];
            shell_M[static_cast<size_t>(i)] += wth * dens_komar;
            shell_Q[static_cast<size_t>(i)] += wth * dens_charge;
        }
    }
    if (matter_ == Matter::Proca) o.lorenz = lor_max / std::max(1e-300, lor_scale_max);
    // R_99 (Siemonsen & East): circular radius (equatorial proper circumference / 2 pi,
    // r sqrt(l/f) at theta = pi/2) of the coordinate sphere containing 99% of the Komar mass
    // (R99_K) or of the charge (R99_Q); cumulative trapezoid in xi, dr = dxi / xi'(r).
    auto r99 = [&](const std::vector<double>& shell) {
        std::vector<double> cum(static_cast<size_t>(Nr_), 0.0);
        auto g = [&](int i) { return (i == Nr_ - 1) ? 0.0 : shell[static_cast<size_t>(i)] / xi_r_[static_cast<size_t>(i)]; };
        for (int i = 1; i < Nr_; ++i) cum[static_cast<size_t>(i)] = cum[static_cast<size_t>(i) - 1] + 0.5 * hr_ * (g(i - 1) + g(i));
        const double target = 0.99 * cum.back();
        int k = 1;
        while (k < Nr_ - 1 && cum[static_cast<size_t>(k)] < target) ++k;
        const double t = (target - cum[static_cast<size_t>(k) - 1]) / std::max(1e-300, cum[static_cast<size_t>(k)] - cum[static_cast<size_t>(k) - 1]);
        const int je = je_;
        auto circ = [&](int i) { return r_[static_cast<size_t>(i)] * std::sqrt(x[idx(i, je, F_l)] / x[idx(i, je, F_f)]); };
        return (1.0 - t) * circ(k - 1) + t * circ(k);
    };
    o.R99_K = r99(shell_M);
    o.R99_Q = r99(shell_Q);

    if (p_.light_rings) {
        // Equatorial profiles at the finite radii r > 0 (origin and infinity excluded).
        const int je = je_;
        std::vector<double> rr, ff, ll, WW;
        for (int i = 1; i + 1 < Nr_; ++i) {
            rr.push_back(r_[static_cast<size_t>(i)]);
            ff.push_back(x[idx(i, je, F_f)]);
            ll.push_back(x[idx(i, je, F_l)]);
            WW.push_back(x[idx(i, je, F_W)]);
        }
        o.lr = find_light_rings(rr, ff, ll, WW);
    }
    if (p_.ergoregion) {
        double gmax = -1e300;
        for (int i = 1; i + 1 < Nr_; ++i)
            for (int j = 0; j < Nt_; ++j) {
                const double sn = std::sin(theta_[static_cast<size_t>(j)]);
                const double f = x[idx(i, j, F_f)], l = x[idx(i, j, F_l)], W = x[idx(i, j, F_W)];
                gmax = std::max(gmax, -f + l * sn * sn * W * W / f);
            }
        std::vector<double> rr, ff, ll, WW;
        for (int i = 1; i + 1 < Nr_; ++i) {
            rr.push_back(r_[static_cast<size_t>(i)]);
            ff.push_back(x[idx(i, je_, F_f)]);
            ll.push_back(x[idx(i, je_, F_l)]);
            WW.push_back(x[idx(i, je_, F_W)]);
        }
        o.ergo = find_ergoregion(rr, ff, ll, WW, gmax);
    }

    // Asymptotic M and J from f = 1 - 2 M G / r + b / r^2 and W = -2 J G / r^2 + d / r^3, fitted
    // at the two outermost finite radii and averaged over theta; G = kappa / (8 pi).
    if (kappa_ > 0.0 && Nr_ > 4) {
        const int i1 = Nr_ - 2, i2 = Nr_ - 3;
        const double r1 = r_[static_cast<size_t>(i1)], r2 = r_[static_cast<size_t>(i2)];
        const std::vector<double> wt = simpson_weights(Nt_, ht_);
        double MG = 0.0, JG2 = 0.0, wsum = 0.0;
        for (int j = 0; j < Nt_; ++j) {
            const double y1 = 1.0 - x[idx(i1, j, F_f)], y2 = 1.0 - x[idx(i2, j, F_f)];
            // y = 2a/r - b/r^2: solve the 2x2 system for a
            const double a = (y1 * r1 * r1 - y2 * r2 * r2) / (2.0 * (r1 - r2));
            const double W1 = x[idx(i1, j, F_W)], W2 = x[idx(i2, j, F_W)];
            // W = c/r^2 + d/r^3: c = (W1 r1^3 - W2 r2^3) / (r1 - r2)
            const double cc = (W1 * r1 * r1 * r1 - W2 * r2 * r2 * r2) / (r1 - r2);
            MG += wt[static_cast<size_t>(j)] * a;
            JG2 += wt[static_cast<size_t>(j)] * cc;
            wsum += wt[static_cast<size_t>(j)];
        }
        const double G = kappa_ / (8.0 * kPi);
        o.mass_asym = MG / wsum / G;
        // Sign: with Phi ~ exp(i(omega t + m varphi)), omega, m > 0, the field pattern rotates toward
        // -varphi and drags frames that way (W < 0), while Kleihaus' J = -int T^t_varphi sqrt(-g) > 0.
        // So W -> -2 J G / r^2 in this convention (checked: |J_asym / J| = 1 to the fit accuracy).
        o.angmom_asym = -JG2 / wsum / (2.0 * G);
    }
    return o;
}

bool RotatingSolver::within(const Vec& R, const Vec& S, double rel) const {
    for (size_t k = 0; k < R.size(); ++k)
        if (std::abs(R[k]) > rel * S[k] + p_.abs_tol_residual) return false;
    return true;
}

bool RotatingSolver::assemble_jacobian(const Vec& x, SparseMatrix& J) {
    const auto t_start = std::chrono::steady_clock::now();
    const size_t n = n_unknowns();
    const size_t last = n - 1;
    const double step_rel = std::cbrt(std::numeric_limits<double>::epsilon());

    double typ[NF_MAX] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < Nr_; ++i)
        for (int j = 0; j < Nt_; ++j)
            for (int a = 0; a < nf_; ++a) typ[a] = std::max(typ[a], std::abs(x[idx(i, j, a)]));
    for (double& t : typ) if (t == 0.0) t = 1.0;

    // Entries per grid row i (parts[i], each sized exactly) plus the omega column and the
    // constraint row (parts[Nr_]); merged in grid order into J, which frees each part as it goes.
    J.clear();
    std::vector<std::vector<SparseLUSolver::Entry>> parts(static_cast<size_t>(Nr_) + 1);
    Vec xp, xm, Rp, Rm;
    // Central differences, one unknown at a time, re-evaluating only the nodes whose rows can
    // depend on it: (i +- 2, j), (i, j +- 2) and the axis row (i, 0) for j <= 4 (one-sided
    // stencil). Parallel over grid rows; each thread perturbs its own copy of the state.
#pragma omp parallel
    {
        Vec xl = x;
        std::vector<SparseLUSolver::Entry> tent;
        double rp[NF_MAX], rm[NF_MAX];
        std::vector<std::pair<int, int>> dep;
#pragma omp for schedule(dynamic, 2) nowait
        for (int i = 0; i < Nr_; ++i) {
            tent.clear();
            for (int j = 0; j < Nt_; ++j) {
                dep.clear();
                if (matter_ != Matter::Proca) {
                    const int reach = (matter_ == Matter::Dirac) ? 3 : 2;  // Dirac: biased stencils
                    for (int di = -reach; di <= reach; ++di)
                        if (i + di >= 0 && i + di < Nr_) dep.emplace_back(i + di, j);
                    for (int dj = -reach; dj <= reach; ++dj)
                        if (dj != 0 && j + dj >= 0 && j + dj < Nt_) dep.emplace_back(i, j + dj);
                } else {
                    // Proca rows contain the mixed derivatives H1_rt, H2_rt: the full 5 x 5 box
                    for (int di = -2; di <= 2; ++di)
                        for (int dj = -2; dj <= 2; ++dj)
                            if (i + di >= 0 && i + di < Nr_ && j + dj >= 0 && j + dj < Nt_)
                                dep.emplace_back(i + di, j + dj);
                }
                {
                    const int reach = (matter_ == Matter::Dirac) ? 3 : 2;
                    if (j > reach && j <= 4) dep.emplace_back(i, 0);                                   // axis rows
                    if (theta_full_ && j < Nt_ - 1 - reach && j >= Nt_ - 5) dep.emplace_back(i, Nt_ - 1);  // pole rows at pi
                }
                for (int a = 0; a < nf_; ++a) {
                    const size_t k = idx(i, j, a);
                    const double xk = x[k];
                    const double h = step_rel * std::max(std::abs(xk), 1e-3 * typ[a]);
                    const double e2 = (xk + h) - (xk - h);
                    for (const auto& d : dep) {
                        xl[k] = xk + h;
                        node_rows(xl, d.first, d.second, rp, nullptr);
                        xl[k] = xk - h;
                        node_rows(xl, d.first, d.second, rm, nullptr);
                        xl[k] = xk;
                        for (int b2 = 0; b2 < nf_; ++b2) {
                            const double v = (rp[b2] - rm[b2]) / e2;
                            if (v != 0.0)
                                tent.push_back({static_cast<int>(idx(d.first, d.second, b2)), static_cast<int>(k), v});
                        }
                    }
                }
            }
            parts[static_cast<size_t>(i)] = tent;  // a copy holds exactly its entries
        }
    }
    std::vector<SparseLUSolver::Entry>& ent = parts[static_cast<size_t>(Nr_)];
    // omega column (dense)
    {
        xp = x;
        xm = x;
        const double h = step_rel * std::max(std::abs(x.back()), 1e-3);
        xp.back() += h;
        xm.back() -= h;
        residual(xp, Rp, nullptr);
        residual(xm, Rm, nullptr);
        for (size_t row = 0; row < last; ++row) {
            const double v = (Rp[row] - Rm[row]) / (2.0 * h);
            if (v != 0.0) ent.push_back({static_cast<int>(row), static_cast<int>(last), v});
        }
    }
    // constraint row, exact
    if (constraint_ == Constraint::Omega) {
        ent.push_back({static_cast<int>(last), static_cast<int>(last), 1.0});
    } else if (constraint_ == Constraint::Arclength) {
        for (size_t k = 0; k < n; ++k)
            if (arc_t_[k] != 0.0) ent.push_back({static_cast<int>(last), static_cast<int>(k), arc_t_[k]});
    } else {
        Vec grad;
        charge_gradient(x, grad);
        for (size_t k = 0; k < n; ++k)
            if (grad[k] != 0.0) ent.push_back({static_cast<int>(last), static_cast<int>(k), grad[k]});
    }

    J.build(static_cast<int>(n), parts);

    if (std::getenv("BS_CHECK_JACOBIAN") && n <= 4000) {
        // Diagnostic: compare the sparse (stencil-local) Jacobian with a column-by-column dense
        // central-difference Jacobian (same steps; constraint row excluded, it is exact).
        std::vector<double> Js(n * n, 0.0), Jd(n * n, 0.0);
        std::vector<SparseLUSolver::Entry> all;
        J.entries(all);
        for (const auto& e : all) Js[static_cast<size_t>(e.row) * n + static_cast<size_t>(e.col)] += e.val;
        for (size_t k = 0; k < n; ++k) {
            xp = x;
            xm = x;
            double h;
            if (k == last) h = step_rel * std::max(std::abs(x.back()), 1e-3);
            else h = step_rel * std::max(std::abs(x[k]), 1e-3 * typ[k % static_cast<size_t>(nf_)]);
            xp[k] += h;
            xm[k] -= h;
            residual(xp, Rp, nullptr);
            residual(xm, Rm, nullptr);
            for (size_t row = 0; row < last; ++row) Jd[row * n + k] = (Rp[row] - Rm[row]) / (xp[k] - xm[k]);
        }
        double maxdiff = 0.0, maxJ = 0.0;
        size_t worst = 0;
        for (size_t row = 0; row < last; ++row)
            for (size_t k = 0; k < n; ++k) {
                const double d = std::abs(Js[row * n + k] - Jd[row * n + k]);
                maxJ = std::max(maxJ, std::abs(Jd[row * n + k]));
                if (d > maxdiff) { maxdiff = d; worst = row * n + k; }
            }
        std::cout << "[check] sparse vs dense Jacobian: max |diff| = " << maxdiff << " (max |J| = " << maxJ
                  << ", worst row " << worst / n << " col " << worst % n << ")\n";
    }

    t_jac_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    return true;
}

bool RotatingSolver::factor_entries(const SparseMatrix& J, int level) {
    if (level < 0 || level >= fallback_levels()) return false;
    if (!lu_) lu_.reset(new SparseLUSolver);
    lu_valid_ = false;
    const auto t_mid = std::chrono::steady_clock::now();
    lu_->set_pivot_threshold(p_.lu_pivot_threshold);
    using Kind = SparseLUSolver::Kind;
    Kind kind = Kind::LU;
    if (precond_ == "ilut") {
        kind = (level < 2) ? Kind::ILUT : Kind::LUFloat;
        lu_->set_ilut(level == 0 ? p_.ilut_fill : 2.0 * p_.ilut_fill, level == 0 ? p_.ilut_droptol : 0.1 * p_.ilut_droptol);
    } else if (precond_ == "lu_float") {
        kind = (level == 0) ? Kind::LUFloat : Kind::LU;
    }
    const bool factored = lu_->factor(J, kind);
    t_lu_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_mid).count();
    if (!factored) {
        std::cout << "[warn] Sparse factorization failed: " << lu_->last_error() << "\n";
        return false;
    }
    lu_valid_ = true;
    lu_constraint_ = constraint_;
    ++n_factorizations_;
    ++n_level_[level];
    return true;
}

bool RotatingSolver::factor_jacobian(const Vec& x) {
    SparseMatrix J;
    return assemble_jacobian(x, J) && factor_entries(J);
}

namespace {
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
    return s;
}
}

bool RotatingSolver::solve_gmres_lu(const SparseMatrix& J, const Vec& b, Vec& x,
                                    int& iterations, int max_iters) const {
    // Right-preconditioned GMRES on the fresh Jacobian with the (stale) factorization as
    // preconditioner: solve (J M^{-1}) u = b, x = M^{-1} u. No restart: gives up after max_iters
    // and lets the caller refactor instead. M is fixed, so x = M^{-1} (V y) costs one more
    // application instead of storing the preconditioned basis.
    const size_t n = b.size();
    const int m = std::max(1, max_iters);
    const double bnorm = std::sqrt(dot(b, b));
    iterations = 0;
    x.assign(n, 0.0);
    if (bnorm == 0.0) return true;
    std::vector<Vec> V(1, b);
    for (double& v : V[0]) v /= bnorm;
    std::vector<std::vector<double>> H(static_cast<size_t>(m) + 1, std::vector<double>(static_cast<size_t>(m), 0.0));
    std::vector<double> cs(static_cast<size_t>(m)), sn(static_cast<size_t>(m)), g(static_cast<size_t>(m) + 1, 0.0);
    g[0] = bnorm;
    Vec z, w;
    int k = 0;
    bool converged = false;
    for (; k < m; ++k) {
        if (!lu_->solve(V[static_cast<size_t>(k)], z)) return false;
        J.multiply(z, w);
        for (int i = 0; i <= k; ++i) {  // modified Gram-Schmidt
            const double h = dot(w, V[static_cast<size_t>(i)]);
            H[static_cast<size_t>(i)][static_cast<size_t>(k)] = h;
            for (size_t q = 0; q < n; ++q) w[q] -= h * V[static_cast<size_t>(i)][q];
        }
        const double hn = std::sqrt(dot(w, w));
        H[static_cast<size_t>(k) + 1][static_cast<size_t>(k)] = hn;
        for (int i = 0; i < k; ++i) {  // apply previous Givens rotations
            const double t = cs[static_cast<size_t>(i)] * H[static_cast<size_t>(i)][static_cast<size_t>(k)]
                             + sn[static_cast<size_t>(i)] * H[static_cast<size_t>(i) + 1][static_cast<size_t>(k)];
            H[static_cast<size_t>(i) + 1][static_cast<size_t>(k)] = -sn[static_cast<size_t>(i)] * H[static_cast<size_t>(i)][static_cast<size_t>(k)]
                                                                    + cs[static_cast<size_t>(i)] * H[static_cast<size_t>(i) + 1][static_cast<size_t>(k)];
            H[static_cast<size_t>(i)][static_cast<size_t>(k)] = t;
        }
        const double a = H[static_cast<size_t>(k)][static_cast<size_t>(k)], bb = H[static_cast<size_t>(k) + 1][static_cast<size_t>(k)];
        const double rr = std::hypot(a, bb);
        cs[static_cast<size_t>(k)] = a / rr;
        sn[static_cast<size_t>(k)] = bb / rr;
        H[static_cast<size_t>(k)][static_cast<size_t>(k)] = rr;
        H[static_cast<size_t>(k) + 1][static_cast<size_t>(k)] = 0.0;
        g[static_cast<size_t>(k) + 1] = -sn[static_cast<size_t>(k)] * g[static_cast<size_t>(k)];
        g[static_cast<size_t>(k)] *= cs[static_cast<size_t>(k)];
        if (std::abs(g[static_cast<size_t>(k) + 1]) <= p_.gmres_lu_tol * bnorm) { ++k; converged = true; break; }
        if (hn == 0.0) { ++k; converged = true; break; }
        V.push_back(w);
        for (double& v : V.back()) v /= hn;
    }
    iterations = k;
    if (!converged) return false;
    std::vector<double> y(static_cast<size_t>(k), 0.0);  // back substitution H y = g
    for (int i = k - 1; i >= 0; --i) {
        double s = g[static_cast<size_t>(i)];
        for (int j = i + 1; j < k; ++j) s -= H[static_cast<size_t>(i)][static_cast<size_t>(j)] * y[static_cast<size_t>(j)];
        y[static_cast<size_t>(i)] = s / H[static_cast<size_t>(i)][static_cast<size_t>(i)];
    }
    Vec u(n, 0.0);
    for (int i = 0; i < k; ++i)
        for (size_t q = 0; q < n; ++q) u[q] += y[static_cast<size_t>(i)] * V[static_cast<size_t>(i)][q];
    return lu_->solve(u, x);
}

bool RotatingSolver::newton_solve(Vec& x) {
    // Newton with Jacobian reuse (chord iterations): the factored Jacobian is kept across
    // iterations and continuation steps and refreshed only when a step reduces the residual by
    // less than chord_rate, needs damping, fails the line search (then retried fresh), or the
    // constraint type changed. Any converged state solves R(x) = 0 regardless of how stale the
    // Jacobian was; staleness only affects the convergence rate.
    Vec R, S, delta, xt, Rt, rhs;
    last_iterations_ = 0;
    last_iters_to_floor_ = 0;
    last_refreshes_ = 0;
    if (lu_valid_ && lu_constraint_ != constraint_) lu_valid_ = false;
    const bool gmres_mode = p_.jacobian_reuse && p_.newton_linear == "gmres_lu";
    const bool chord_mode = p_.jacobian_reuse && !gmres_mode;
    SparseMatrix jent;  // the fresh Jacobian (shared by GMRES and the factorization)
    bool refresh = !p_.jacobian_reuse || !lu_valid_;
    const int max_its = chord_mode ? 3 * p_.max_newton_iters : p_.max_newton_iters;
    for (int it = 0; it < max_its; ++it) {
        residual(x, R, &S);
        double rn = 0.0;
        for (double v : R) rn = std::max(rn, std::abs(v));
        last_residual_ = rn;
        last_iterations_ = it + 1;
        ++n_newton_its_;
        if (std::getenv("BS_NEWTON_VERBOSE")) {
            // per-field max |residual| / scale and the omega row, with the node of the worst field
            double mx[NF_MAX] = {0}, worst = 0.0;
            size_t wk = 0, mk[NF_MAX] = {0};
            for (size_t k = 0; k + 1 < R.size(); ++k) {
                const double v = std::abs(R[k]) / (S[k] + 1e-300);
                const int a = static_cast<int>(k % static_cast<size_t>(nf_));
                if (std::abs(R[k]) > mx[a]) { mx[a] = std::abs(R[k]); mk[a] = k / static_cast<size_t>(nf_); }
                if (v > worst && std::abs(R[k]) > 1e-12) { worst = v; wk = k; }
            }
            std::cout << "[newton] it " << it << " argmax|R| (i,j) per field:";
            for (int a = 0; a < nf_; ++a) std::cout << " (" << mk[a] / static_cast<size_t>(Nt_) << "," << mk[a] % static_cast<size_t>(Nt_) << ")";
            std::cout << "\n";
            std::cout << "[newton] it " << it << " max|R| per field:";
            for (int a = 0; a < nf_; ++a) std::cout << ' ' << mx[a];
            const size_t node = wk / static_cast<size_t>(nf_);
            std::cout << " | constraint " << R.back() << " | worst rel " << worst << " at i=" << node / static_cast<size_t>(Nt_)
                      << " j=" << node % static_cast<size_t>(Nt_) << " field " << wk % static_cast<size_t>(nf_) << "\n";
        }
        const bool at_floor = within(R, S, p_.stall_rel_tol);
        if (at_floor && last_iters_to_floor_ == 0) last_iters_to_floor_ = it + 1;
        if (within(R, S, p_.rel_tol_residual)) break;

        rhs.resize(R.size());
        for (size_t k = 0; k < R.size(); ++k) rhs[k] = -R[k];
        bool fresh;
        if (gmres_mode) {
            // Newton step with the fresh Jacobian, solved by GMRES preconditioned with the stale
            // LU; refactor only when GMRES does not converge within gmres_lu_max_iters.
            fresh = true;
            if (!assemble_jacobian(x, jent)) return false;
            int its = 0;
            const int m = std::max(1, p_.gmres_lu_max_iters);
            bool solved = lu_valid_ && solve_gmres_lu(jent, rhs, delta, its, m);
            n_gmres_its_ += its;
            if (!solved) {
                ++last_refreshes_;
                // Fresh factorization, climbing the fallback ladder when one breaks down or (if
                // inexact) does not get GMRES to converge within 3 x gmres_lu_max_iters.
                for (int level = 0; level < fallback_levels() && !solved; ++level) {
                    if (!factor_entries(jent, level)) continue;
                    if (lu_->exact()) {
                        if (!lu_->solve(rhs, delta)) return false;
                        solved = true;
                    } else {
                        solved = solve_gmres_lu(jent, rhs, delta, its, 3 * m);
                        n_gmres_its_ += its;
                    }
                }
                if (!solved) return false;
            }
        } else {
            fresh = refresh || !lu_valid_;
            if (fresh) {
                if (!factor_jacobian(x)) return false;
                ++last_refreshes_;
            }
            if (!lu_->solve(rhs, delta)) return false;
        }

        double lambda = 1.0, rt = 0.0;
        bool accepted = false;
        for (int bt = 0; bt < p_.max_line_search_steps; ++bt) {
            xt = x;
            for (size_t k = 0; k < x.size(); ++k) xt[k] += lambda * delta[k];
            for (int i = 0; i < Nr_; ++i)
                for (int j = 0; j < Nt_; ++j)
                    for (int a : {F_f, F_l, F_g}) xt[idx(i, j, a)] = std::max(1e-8, xt[idx(i, j, a)]);
            residual(xt, Rt, nullptr);
            rt = 0.0;
            for (double v : Rt) rt = std::max(rt, std::abs(v));
            if (rt <= (1.0 - p_.armijo_c * lambda) * rn) { accepted = true; break; }
            lambda *= 0.5;
            if (lambda < p_.line_search_min || (!fresh && bt >= 2)) break; // stale: refresh instead
        }
        if (!accepted) {
            if (!fresh) { refresh = true; continue; } // retry this iterate with a fresh Jacobian
            if (it > 0 && at_floor) break;           // round-off floor
            std::cout << "[warn] Line search failed (residual " << rn << ").\n";
            return false;
        }
        x = xt;
        refresh = !chord_mode || lambda < 1.0 || rt > p_.chord_rate * rn;
        double dn = 0.0;
        for (double v : delta) dn = std::max(dn, lambda * std::abs(v));
        if (std::getenv("BS_NEWTON_VERBOSE")) {
            size_t km = 0;
            for (size_t k = 0; k + 1 < delta.size(); ++k) if (std::abs(delta[k]) > std::abs(delta[km])) km = k;
            const size_t node = km / static_cast<size_t>(nf_);
            std::cout << "[newton]   step: lambda " << lambda << ", max|dx| " << dn << " (full step " << std::abs(delta[km])
                      << " at i=" << node / static_cast<size_t>(Nt_) << " j=" << node % static_cast<size_t>(Nt_) << " field "
                      << km % static_cast<size_t>(nf_) << "), |R| " << rn << " -> " << rt << "\n";
        }
        if (dn < p_.abs_tol_update) {
            if (!at_floor) return false;
            break;
        }
        if (it == max_its - 1) return false;
    }
    double pmax = 0.0;
    for (int i = 0; i < Nr_; ++i)
        for (int j = 0; j < Nt_; ++j) pmax = std::max(pmax, std::abs(x[idx(i, j, F_p)]));
    if (pmax < p_.trivial_phi_tol) {
        std::cout << "[warn] Converged to the trivial vacuum (max|phi| = " << pmax << ").\n";
        return false;
    }
    return true;
}

void RotatingSolver::write_profile(const std::string& path, const Vec& x) const {
    std::ofstream out(path);
    out << "# rotating star profile: matter=" << p_.matter << " potential=" << p_.potential << " m=" << m_ << " kappa=" << kappa_
        << " omega=" << std::setprecision(12) << x.back() << "\n";
    out << "# radial map xi(r): L=" << map_.L;
    for (int k = 0; k < 2; ++k) {
        const RadialMap::Cluster& cl = map_.cl[static_cast<size_t>(k)];
        out << " cluster" << k << "(fraction=" << map_.fraction(k) << " ra=" << cl.ra << " rb=" << cl.rb << " delta=" << cl.delta << ")";
    }
    out << "\n";
    out << "# i j r theta f l g W "
        << (matter_ == Matter::Scalar ? "phi" : matter_ == Matter::Proca ? "V H1 H2 H3" : "D1 D2 D3 D4") << "\n";
    out << std::scientific << std::setprecision(10);
    for (int i = 0; i + 1 < Nr_; ++i)
        for (int j = 0; j < Nt_; ++j) {
            out << i << ' ' << j << ' ' << r_[static_cast<size_t>(i)] << ' ' << theta_[static_cast<size_t>(j)];
            for (int a = 0; a < nf_; ++a) out << ' ' << x[idx(i, j, a)];
            out << '\n';
        }
}

void RotatingSolver::append_family(const std::string& path, const Vec& x, bool header) const {
    std::ofstream out(path, header ? std::ios::trunc : std::ios::app);
    if (header) {
        out << "# rotating family: matter=" << p_.matter << " potential=" << p_.potential << " m=" << m_ << " Nr=" << Nr_
            << " Ntheta=" << Nt_ << "\n";
        // phi_max: max of the matter amplitude (|phi|; Proca sqrt(V^2 + H1^2 + H2^2 + H3^2))
        out << "# kappa omega phi_max mass_komar mass_asym charge angmom angmom_asym residual_inf newton_iters"
               " J_over_M2 R99_K R99_Q C_K C_Q"
            << (p_.light_rings ? " light_rings" : "") << (p_.ergoregion ? " ergo_gtt_max" : "")
            << (matter_ == Matter::Proca ? " lorenz" : "") << "\n";
    }
    const Observables o = observables(x);
    out << std::scientific << std::setprecision(12) << kappa_ << ' ' << o.omega << ' ' << o.phi_max << ' '
        << o.mass_komar << ' ' << o.mass_asym << ' ' << o.charge << ' ' << o.angmom << ' ' << o.angmom_asym << ' '
        << last_residual_ << ' ' << last_iterations_ << ' ' << o.angmom / (o.mass_komar * o.mass_komar) << ' '
        << o.R99_K << ' ' << o.R99_Q << ' ' << o.mass_komar / o.R99_K << ' ' << o.mass_komar / o.R99_Q;
    if (p_.light_rings) {
        out << ' ' << o.lr.count();
        if (o.lr.count() > 0) {
            std::cout << "[info]   light rings (circumferential radius / M):";
            for (double R : o.lr.R_co) std::cout << " co-rotating " << R / o.mass_komar;
            for (double R : o.lr.R_counter) std::cout << " counter-rotating " << R / o.mass_komar;
            std::cout << "\n";
        }
    }
    if (p_.ergoregion) {
        out << ' ' << o.ergo.gtt_max;
        if (o.ergo.present()) {
            std::cout << "[info]   ergoregion: max g_tt " << o.ergo.gtt_max << "; equatorial ergosurface (circumferential radius / M):";
            for (double R : o.ergo.R) std::cout << ' ' << R / o.mass_komar;
            std::cout << "\n";
        }
    }
    if (matter_ == Matter::Proca) out << ' ' << o.lorenz;
    out << '\n';
}

bool RotatingSolver::run() {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = run_impl();
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "[info] Newton iterations " << n_newton_its_ << ", GMRES iterations " << n_gmres_its_
              << ", Jacobian factorizations: " << n_factorizations_;
    if (precond_ == "ilut")
        std::cout << " (fallbacks: " << n_level_[1] << " to a stronger ILUT, " << n_level_[2] << " to float LU)";
    if (precond_ == "lu_float") std::cout << " (fallbacks: " << n_level_[1] << " to exact LU)";
    std::cout << ", regrids: " << n_regrids_ << "; time: total " << total
              << " s, Jacobian (finite differences) " << t_jac_ << " s, LU factorization " << t_lu_
              << " s, other (residuals, solves, I/O) " << total - t_jac_ - t_lu_ << " s\n";
    return ok;
}

bool RotatingSolver::run_impl() {
    if (Nr_ < 6 || Nt_ < 6) {
        std::cout << "[error] Rotating solver needs Nr >= 6 and Ntheta >= 6.\n";
        return false;
    }
    if (p_.linear_preconditioner != "lu" && p_.linear_preconditioner != "lu_float" && p_.linear_preconditioner != "ilut") {
        std::cout << "[error] linear_preconditioner must be lu, lu_float or ilut.\n";
        return false;
    }
    precond_ = p_.linear_preconditioner;
    if (precond_ != "lu" && !(p_.jacobian_reuse && p_.newton_linear == "gmres_lu")) {
        // Chord and plain Newton use the factorization as the Newton step: it must be exact.
        std::cout << "[warn] linear_preconditioner = " << precond_
                  << " needs jacobian_reuse = 1 and newton_linear = gmres_lu; using the exact LU.\n";
        precond_ = "lu";
    }
    if (p_.radial_grid != "compactified" && p_.radial_grid != "adaptive") {
        std::cout << "[error] radial_grid must be compactified or adaptive.\n";
        return false;
    }
    map_ = RadialMap();
    if (p_.matter != "scalar" && p_.matter != "proca" && p_.matter != "dirac") {
        std::cout << "[error] matter must be scalar, proca or dirac.\n";
        return false;
    }
    setup_matter();
    map_.L = (p_.radial_scale > 0.0) ? p_.radial_scale : 1.0;
    build_grid();
    seed_analytic();
    if (!p_.init_file.empty() && p_.radial_grid == "adaptive") {
        if (init_map_valid_) {
            // the profile's own radial map: with the same Nr the nodes coincide (no radial interpolation)
            map_ = init_map_;
            map_adapted_ = true;
            build_grid();
            load_init_file();
        } else {
            // fit the adaptive map to the loaded profile before the first solve (twice: the first fit
            // sees the profile on the default grid, the second on the fitted one)
            for (int pass = 0; pass < 2; ++pass) {
                map_ = wanted_map(x_);
                build_grid();
                load_init_file();
            }
        }
    }
    const std::string fam = "families2d.dat";
    auto report = [&](const char* tag) {
        const Observables o = observables(x_);
        std::cout << "[info] " << tag << ": kappa = " << kappa_ << "  omega = " << o.omega << "  phi_max = " << o.phi_max
                  << "  M = " << o.mass_komar << "  Q = " << o.charge << "  J = " << o.angmom
                  << "  (Newton its " << last_iterations_ << ", |R| " << last_residual_ << ")\n";
    };

    // 1. Flat space (kappa = 0) solution at omega_target from the analytic seed. Without
    //    flat-space Q-balls (mini, repulsive) the kappa = 0 problem is linear in phi and only
    //    phi = 0 solves it, so solve directly at kappa_target instead (as in the 1D solver).
    constraint_ = Constraint::Omega;
    omega_target_ = p_.omega_target;
    const bool qballs = matter_ == Matter::Scalar && potential_has_qballs(p_);  // free Proca: none
    kappa_ = qballs ? 0.0 : std::max(0.0, p_.kappa_target);
    if (!newton_solve(x_)) {
        std::cout << "[warn] Initial solve at kappa = " << kappa_ << " failed.\n";
        write_profile("bs2d.dat", x_);
        return false;
    }
    report(kappa_ == 0.0 ? "Q-ball" : "initial star");
    // Adaptive grid at fixed omega and kappa: refit the map to x_ and re-solve until it settles.
    auto adapt_fixed = [&]() {
        for (int pass = 0; pass < 3; ++pass) {
            const RadialMap want = wanted_map(x_);
            if (!map_drifted(want) || !regrid(want, {&x_}, [&]() { return newton_solve(x_); })) break;
        }
    };
    adapt_fixed();

    // 2. Ramp the gravitational coupling at fixed omega.
    const double kappa_target = std::max(0.0, p_.kappa_target);
    while (kappa_ < kappa_target - 1e-14) {
        const Vec x_save = x_;
        const double k_save = kappa_;
        double dk = std::min(p_.delta_kappa, kappa_target - kappa_);
        bool ok = false;
        while (!ok && dk > 1e-6) {
            kappa_ = k_save + dk;
            x_ = x_save;
            ok = newton_solve(x_);
            if (!ok) dk *= 0.5;
        }
        if (!ok) {
            std::cout << "[warn] kappa ramp failed at kappa = " << k_save << ".\n";
            kappa_ = k_save;
            x_ = x_save;
            break;
        }
        adapt_fixed();
        report("kappa ramp");
    }
    append_family(fam, x_, true);
    write_profile("bs2d.dat", x_);

    if (p_.continuation_mode == "arclength") return run_arclength(fam);

    // 3. March in omega (secant predictor). If charge_target > 0, stop at the first
    //    crossing of Q = charge_target and solve there with the fixed-Q constraint.
    if (p_.delta_omega <= 0.0) return true;
    Vec x_prev = x_;
    double Q_prev = charge(x_);
    double dw = p_.delta_omega;
    double w_last_step = 0.0;
    // The last step is shortened to land exactly on rot_omega_stop.
    for (int step = 0; step < p_.rot_max_steps && x_.back() > p_.rot_omega_stop + 1e-12; ) {
        const Vec x_cur = x_;
        const double w_cur = x_cur.back();
        const double dw_eff = std::min(dw, w_cur - p_.rot_omega_stop);
        omega_target_ = w_cur - dw_eff;
        Vec seed = x_cur;
        if (w_last_step != 0.0) {
            const double s = dw_eff / w_last_step;
            for (size_t k = 0; k < seed.size(); ++k) seed[k] += s * (x_cur[k] - x_prev[k]);
        }
        seed.back() = omega_target_;
        x_ = seed;
        if (!newton_solve(x_)) {
            x_ = x_cur;
            dw *= 0.5;
            if (dw < 1e-5) { std::cout << "[warn] omega march stalled.\n"; break; }
            continue;
        }
        ++step;
        x_prev = x_cur;
        w_last_step = dw_eff;
        if (p_.radial_grid == "adaptive") {
            const RadialMap want = wanted_map(x_);
            if (map_drifted(want)) regrid(want, {&x_, &x_prev}, [&]() { return newton_solve(x_); });
        }
        append_family(fam, x_, false);
        report("omega step");
        const double Q = charge(x_);
        if (p_.charge_target > 0.0 && matter_ == Matter::Scalar &&  // fixed-Q rows: scalar only
            (Q - p_.charge_target) * (Q_prev - p_.charge_target) <= 0.0) {
            // Fixed-Q solve seeded by linear interpolation between the bracketing models.
            const double t = (p_.charge_target - Q_prev) / (Q - Q_prev);
            Vec xq = x_prev;
            for (size_t k = 0; k < xq.size(); ++k) xq[k] += t * (x_[k] - x_prev[k]);
            constraint_ = Constraint::Charge;
            charge_target_ = p_.charge_target;
            if (newton_solve(xq)) {
                x_ = xq;
                report("fixed-Q model");
                append_family(fam, x_, false);
                write_profile("bs2d.dat", x_);
            } else {
                std::cout << "[warn] fixed-Q solve failed.\n";
            }
            return true;
        }
        Q_prev = Q;
        if (step % 5 == 0) write_profile("bs2d.dat", x_);
    }
    write_profile("bs2d.dat", x_);
    return true;
}

void RotatingSolver::resolution_indicators(const Vec& x, double& core, double& wall) const {
    core = 0.0;
    double pmax = 0.0, jump = 0.0;
    for (int j = 0; j < Nt_; ++j)
        core = std::max(core, x[idx(1, j, F_f)] / std::max(1e-300, x[idx(0, j, F_f)]) - 1.0);
    for (int i = 0; i < Nr_; ++i)
        for (int j = 0; j < Nt_; ++j) {
            const double v = amp(x, i, j);
            pmax = std::max(pmax, v);
            if (i + 1 < Nr_) jump = std::max(jump, std::abs(amp(x, i + 1, j) - v));
            if (j + 1 < Nt_) jump = std::max(jump, std::abs(amp(x, i, j + 1) - v));
        }
    wall = jump / std::max(1e-300, pmax);
}

bool RotatingSolver::run_arclength(const std::string& fam) {
    // Same scheme as BosonStarSolver::run_arclength_continuation: with x_prev, x_cur the last
    // two solutions, t = secant, seed x_pred = x_cur + ds t, constraint <t, x - x_pred> = 0.
    auto norm = [](const Vec& v) { double s = 0.0; for (double a : v) s += a * a; return std::sqrt(s); };
    auto report = [&](const char* tag) {
        const Observables o = observables(x_);
        std::cout << "[info] " << tag << ": omega = " << o.omega << "  phi_max = " << o.phi_max
                  << "  M = " << o.mass_komar << "  Q = " << o.charge << "  J = " << o.angmom
                  << "  J/M^2 = " << o.angmom / (o.mass_komar * o.mass_komar) << "  C_K = " << o.mass_komar / o.R99_K << "\n";
    };
    Vec x_prev = x_;
    const double w0 = x_.back();
    double dw = (p_.delta_omega != 0.0) ? p_.delta_omega : 1e-3; // sign sets the initial direction
    // A restart (init_file) may sit next to a fold, where a fixed-omega step in the continuation
    // direction can fail or land off the branch: step backwards instead (the family is single
    // valued on the side it came from) and swap the pair, so the secant points forward.
    const bool back_start = !p_.init_file.empty();
    if (back_start) dw = -dw;
    bool ok = false;
    for (int tries = 0; tries < 8 && !ok; ++tries) {
        constraint_ = Constraint::Omega;
        omega_target_ = w0 - dw;
        x_ = x_prev;
        x_.back() = omega_target_;
        ok = newton_solve(x_);
        if (!ok) dw *= 0.5;
    }
    if (!ok) { std::cout << "[warn] Arclength: initial omega step failed.\n"; x_ = x_prev; return false; }
    if (back_start) std::swap(x_, x_prev);  // x_prev: the backward step; x_: the restart profile
    else append_family(fam, x_, false);
    report("Arclength start");

    Vec x_cur = x_;
    Observables o_cur = observables(x_cur);
    int first_law_retries = 0;
    Vec x_prev2;          // third-to-last solution (quadratic predictor)
    double h_prev = 0.0;  // |x_prev - x_prev2|
    Vec sec(x_cur.size());
    for (size_t k = 0; k < sec.size(); ++k) sec[k] = x_cur[k] - x_prev[k];
    double ds = norm(sec);
    const double ds_max = (p_.arclength_ds_max > 0.0) ? p_.arclength_ds_max : 10.0 * ds;
    const double ds_min = 1e-6 * ds;
    ds = std::min(ds, ds_max);
    const double mB = (matter_ == Matter::Scalar) ? boson_mass(p_) : p_.mu;
    constraint_ = Constraint::Arclength;
    int accepted = 0;
    while (accepted < p_.arclength_max_steps) {
        for (size_t k = 0; k < sec.size(); ++k) sec[k] = x_cur[k] - x_prev[k];
        const double sn = norm(sec);
        arc_t_.resize(sec.size());
        arc_pred_.resize(sec.size());
        for (size_t k = 0; k < sec.size(); ++k) {
            arc_t_[k] = sec[k] / sn;
            arc_pred_[k] = x_cur[k] + ds * arc_t_[k];
        }
        // Newton seed: the linear predictor, or a quadratic extrapolation through the last three
        // solutions (in chord length) -- a better starting point on curved branches. The seed
        // does not change the solution, which the arclength constraint fixes.
        if (p_.arclength_predictor == "quadratic" && !x_prev2.empty()) {
            const double h2 = sn, s1 = -h2, s2 = -(h2 + h_prev);
            const double sd = ds;
            const double w0 = (sd - s1) * (sd - s2) / (s1 * s2);
            const double w1 = sd * (sd - s2) / (s1 * (s1 - s2));
            const double w2 = sd * (sd - s1) / (s2 * (s2 - s1));
            x_.resize(x_cur.size());
            for (size_t k = 0; k < x_.size(); ++k) x_[k] = w0 * x_cur[k] + w1 * x_prev[k] + w2 * x_prev2[k];
        } else {
            x_ = arc_pred_;
        }
        bool step_ok = newton_solve(x_);
        const int iters = (last_iters_to_floor_ > 0) ? last_iters_to_floor_ : last_iterations_;
        const int refreshes = last_refreshes_;
        if (step_ok) {
            double dot = 0.0, dn = 0.0;
            for (size_t k = 0; k < x_.size(); ++k) {
                const double d = x_[k] - x_cur[k];
                dot += d * arc_t_[k];
                dn += d * d;
            }
            const double cosang = dot / std::max(1e-300, std::sqrt(dn));
            if (cosang < p_.arclength_min_cos) {
                std::cout << "[warn] Arclength: secant turned too sharply (cos = " << cosang << "); reducing ds.\n";
                step_ok = false;
            }
        }
        if (!step_ok) {
            ds *= 0.5;
            if (ds < ds_min) { std::cout << "[warn] Arclength: step size underflow; stopping.\n"; x_ = x_cur; return false; }
            continue;
        }
        double core, wall;
        resolution_indicators(x_, core, wall);
        if (core > p_.core_resolution_tol || wall > p_.wall_resolution_tol) {
            std::cout << "[info] Arclength: profile under-resolved (core " << core << ", wall " << wall
                      << "; tolerances " << p_.core_resolution_tol << ", " << p_.wall_resolution_tol
                      << "); model discarded, stopping. Increase Nr/Ntheta to continue.\n";
            x_ = x_cur;
            return true;
        }
        // First law dM = omega dQ between consecutive models (midpoint rule, error O(h^3)). The
        // Q |d omega| term keeps the measure finite at cusps, where dM and dQ vanish together.
        // Resolved stretches give ~1e-4 and grid noise (a thin wall crossing cells) a few 1e-2;
        // jumps onto spurious discrete branches and their fake turning points give 0.05-1. Not
        // retried with a shorter step: that only amplifies the noise (~ noise / step).
        const Observables o_new = observables(x_);
        if (p_.first_law_tol > 0.0) {
            const double dM = o_new.mass_komar - o_cur.mass_komar, dQ = o_new.charge - o_cur.charge;
            const double wm = 0.5 * (o_new.omega + o_cur.omega), Qm = 0.5 * (o_new.charge + o_cur.charge);
            const double dw = o_new.omega - o_cur.omega;
            const double viol = std::abs(dM - wm * dQ) /
                                std::max(1e-300, std::abs(dM) + wm * std::abs(dQ) + Qm * std::abs(dw));
            if (viol > p_.first_law_tol) {
                // The midpoint rule's own error grows with the step on a sharply curving branch:
                // retry shorter first (this can only postpone a stop, never cause one).
                if (first_law_retries < p_.first_law_retries) {
                    ++first_law_retries;
                    std::cout << "[warn] Arclength: first law violated by " << viol << "; retrying with ds / 2.\n";
                    ds *= 0.5;
                    continue;
                }
                std::cout << "[info] Arclength: first law dM = omega dQ violated by " << viol << " (tolerance "
                          << p_.first_law_tol << "); model discarded, stopping. Increase Nr/Ntheta to continue.\n";
                x_ = x_cur;
                return true;
            }
        }
        first_law_retries = 0;
        o_cur = o_new;
        x_prev2 = x_prev;
        h_prev = sn;
        x_prev = x_cur;
        x_cur = x_;
        ++accepted;
        append_family(fam, x_, false);
        report("Arclength step");
        std::cout << "[info]   ds = " << ds << " (ds_max " << ds_max << "), Newton iterations " << iters
                  << ", Jacobian refreshes " << refreshes << "\n";
        // every accepted model with write_bsdata_per_step (restart points via init_file), else every 10th
        if (p_.write_bsdata_per_step || accepted % 10 == 0) write_profile("bs2d.dat", x_);
        if (p_.radial_grid == "adaptive") {
            const RadialMap want = wanted_map(x_cur);
            if (map_drifted(want)) {
                // Re-converge both secant ends on the new grid, each with the arclength constraint
                // through its interpolated state along the (interpolated) secant: well posed at folds.
                auto resolve = [&]() {
                    Vec t(x_cur.size());
                    for (size_t k = 0; k < t.size(); ++k) t[k] = x_cur[k] - x_prev[k];
                    const double tn = norm(t);
                    for (double& v : t) v /= tn;
                    for (Vec* X : {&x_cur, &x_prev}) {
                        arc_t_ = t;
                        arc_pred_ = *X;
                        x_ = *X;
                        if (!newton_solve(x_)) return false;
                        *X = x_;
                    }
                    return true;
                };
                if (regrid(want, {&x_cur, &x_prev}, resolve)) {
                    x_prev2.clear();
                    o_cur = observables(x_cur);
                }
                x_ = x_cur;
            }
        }
        const double w = x_.back();
        const double w_max = (p_.arclength_omega_max > 0.0) ? std::min(mB, p_.arclength_omega_max) : mB;
        if (w <= p_.arclength_omega_min || w >= w_max) {
            std::cout << "[info] Arclength: omega = " << w << " left (" << p_.arclength_omega_min << ", " << w_max
                      << "); stopping.\n";
            break;
        }
        // Step-size control from the Newton effort. Newton (also with GMRES + stale-LU
        // preconditioner): iterations to the round-off floor. Chord iterations are cheap and
        // plentiful; there the number of fresh factorizations measures how far the predictor
        // landed from the solution (0-1: well inside the basin, >= 3: struggling).
        const bool chord = p_.jacobian_reuse && p_.newton_linear != "gmres_lu";
        const bool easy = chord ? (refreshes <= 1) : (iters <= 3);
        const bool hard = chord ? (refreshes >= 3) : (iters >= 7);
        if (easy) ds = std::min(ds_max, 1.5 * ds);
        else if (hard) ds *= 0.7;
    }
    write_profile("bs2d.dat", x_);
    return true;
}

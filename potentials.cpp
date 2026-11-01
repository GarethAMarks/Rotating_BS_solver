#include "potentials.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.1415926535897932384626433832795;

void kkls_coeffs(const BSParams& p, double& c1, double& c2) {
    const double kk = std::max(1e-14, p.kkls_kappa);
    c1 = 16.0 * kPi / (1.1 * kk);
    c2 = 64.0 * kPi * kPi / (1.1 * kk * kk);
}
}

double potential_U(const BSParams& p, double phi) {
    const double mu2 = p.mu * p.mu;
    const double x = phi * phi;

    if (p.potential == "repulsive") {
        return mu2 * x + p.lambda_rep * x * x;
    }
    if (p.potential == "solitonic") {
        const double sigma2 = std::max(1e-14, p.sigma * p.sigma);
        const double a = 1.0 - 2.0 * x / sigma2;
        return mu2 * x * a * a;
    }
    if (p.potential == "kkls") {
        double c1, c2;
        kkls_coeffs(p, c1, c2);
        return mu2 * x * (1.0 - c1 * x + c2 * x * x);
    }
    if (p.potential == "axionic") {
        const double f = std::max(1e-12, p.f_axion);
        return mu2 * f * f * (1.0 - std::cos(std::sqrt(2.0) * phi / f));
    }
    if (p.potential == "kleihaus") {
        return p.kleihaus_lambda * x * (x * x - p.kleihaus_a * x + p.kleihaus_b);
    }
    return mu2 * x; // mini
}

double potential_dU(const BSParams& p, double phi) {
    const double mu2 = p.mu * p.mu;
    const double x = phi * phi;

    if (p.potential == "repulsive") {
        return 2.0 * mu2 * phi + 4.0 * p.lambda_rep * phi * x;
    }
    if (p.potential == "solitonic") {
        const double sigma2 = std::max(1e-14, p.sigma * p.sigma);
        const double a = 1.0 - 2.0 * x / sigma2;
        return 2.0 * mu2 * phi * a * a - 8.0 * mu2 * phi * x * a / sigma2;
    }
    if (p.potential == "kkls") {
        double c1, c2;
        kkls_coeffs(p, c1, c2);
        return 2.0 * mu2 * phi - 4.0 * mu2 * c1 * phi * x + 6.0 * mu2 * c2 * phi * x * x;
    }
    if (p.potential == "axionic") {
        const double f = std::max(1e-12, p.f_axion);
        const double s2 = std::sqrt(2.0);
        return mu2 * f * s2 * std::sin(s2 * phi / f);
    }
    if (p.potential == "kleihaus") {
        return p.kleihaus_lambda * phi * (6.0 * x * x - 4.0 * p.kleihaus_a * x + 2.0 * p.kleihaus_b);
    }
    return 2.0 * mu2 * phi;
}

double potential_dU_abs_terms(const BSParams& p, double phi) {
    const double mu2 = p.mu * p.mu;
    const double a = std::abs(phi);
    const double x = phi * phi;
    if (p.potential == "repulsive") return 2.0 * mu2 * a + 4.0 * std::abs(p.lambda_rep) * a * x;
    if (p.potential == "solitonic") {
        const double sigma2 = std::max(1e-14, p.sigma * p.sigma);
        const double b = 1.0 - 2.0 * x / sigma2;
        return 2.0 * mu2 * a * b * b + 8.0 * mu2 * a * x * std::abs(b) / sigma2;
    }
    if (p.potential == "kkls") {
        double c1, c2;
        kkls_coeffs(p, c1, c2);
        return 2.0 * mu2 * a + 4.0 * mu2 * c1 * a * x + 6.0 * mu2 * c2 * a * x * x;
    }
    if (p.potential == "kleihaus") {
        return std::abs(p.kleihaus_lambda) * a *
               (6.0 * x * x + 4.0 * std::abs(p.kleihaus_a) * x + 2.0 * std::abs(p.kleihaus_b));
    }
    return std::abs(potential_dU(p, phi));
}

double boson_mass(const BSParams& p) {
    if (p.potential == "kleihaus") return std::sqrt(std::max(0.0, p.kleihaus_lambda * p.kleihaus_b));
    return p.mu;
}

bool potential_has_qballs(const BSParams& p) {
    return p.potential == "kkls" || p.potential == "solitonic" || p.potential == "axionic" ||
           p.potential == "kleihaus";
}

#ifndef POTENTIALS_H_
#define POTENTIALS_H_

#include "bs_params.h"

// Scalar self-interaction potentials U(phi), phi = |Phi| (Kleihaus et al. convention:
// action -dPhi* dPhi - U, field equation uses dU/dphi). Shared by the spherical and
// rotating solvers.
//   mini       U = mu^2 phi^2
//   repulsive  U = mu^2 phi^2 + lambda_rep phi^4
//   solitonic  U = mu^2 phi^2 (1 - 2 phi^2 / sigma^2)^2                    (S&E Eq. 2)
//   kkls       U = mu^2 phi^2 [1 - 16 pi/(1.1 k) phi^2 + 64 pi^2/(1.1 k^2) phi^4]  (S&E Eq. 3)
//   axionic    U = mu^2 f^2 (1 - cos(sqrt(2) phi / f))                     (S&E Eq. 4)
//   kleihaus   U = lambda (phi^6 - a phi^4 + b phi^2)                       (Kleihaus et al. Eq. 2)
double potential_U(const BSParams& p, double phi);
double potential_dU(const BSParams& p, double phi);

// Sum of |terms| of dU/dphi: the rounding-error scale of potential_dU, which can
// itself vanish by cancellation (e.g. on a solitonic plateau).
double potential_dU_abs_terms(const BSParams& p, double phi);

// Boson mass m_B = sqrt(U''(0) / 2): mu, or sqrt(lambda b) for the kleihaus potential.
// Bound states need omega < m_B.
double boson_mass(const BSParams& p);

// Potentials with an attractive self-interaction admit flat-space Q-balls.
bool potential_has_qballs(const BSParams& p);

#endif // POTENTIALS_H_

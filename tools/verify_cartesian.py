"""Verify the Cartesian 3+1 expressions written by output_cartesian_values (rotating solver).

Metric (as derive_rotating.py):
    ds^2 = -f dt^2 + (l/f) [ g (dr^2 + r^2 dtheta^2) + r^2 sin^2(theta) (dvarphi - W/r dt)^2 ]
Scalar Phi = phi(r, theta) exp(i (w t + m varphi)), T_ab = dPhi*_a dPhi_b + c.c. - g_ab (g^cd dPhi*_c dPhi_d + U),
G_ab = kappa T_ab.  x = r sin(theta) cos(varphi), y = r sin(theta) sin(varphi), z = r cos(theta).

Closed forms (as implemented in RotatingSolver::write_cartesian), with
    n = (s cp, s sp, c),  th = (c cp, c sp, -s),  e = (-sp, cp, 0)   (s = sin theta, cp = cos varphi, ...):
    alpha    = sqrt(f)
    beta^i   = -W s e^i
    gamma_ij = A delta_ij + (B - A) e_i e_j,   A = l g / f,  B = l / f
    K_ij     = a (n_i e_j + e_i n_j) + b (th_i e_j + e_i th_j),
               a = -(l s / (2 f^(3/2))) (W_r - W/r),  b = -(l s / (2 f^(3/2))) W_theta / r
               convention K_ij = -(1/(2 alpha)) (d_t gamma_ij - D_i beta_j - D_j beta_i)
    d_i Phi  = e^{i m varphi} [ phi_r n_i + (phi_theta / r) th_i + i m q e_i ],  q = phi / (r s)
    Pi       = n^a d_a Phi = (d_t Phi - beta^i d_i Phi) / alpha = i (w + m W/r) Phi / sqrt(f)
    rho      = n^a n^b T_ab = (w + m W/r)^2 phi^2 / f + (phi_r^2 + (phi_theta/r)^2) / A + m^2 q^2 / B + U
    S_i      = -gamma_i^a n^b T_ab = -2 m (w + m W/r) phi q / sqrt(f) e_i

Check 1: every closed form against the same quantity computed from the 4-metric written in Cartesian
         coordinates (pullback of the line element, inverse 4-metric -> alpha, beta; K from the Lie
         derivative with partial derivatives; T_ab from the 4D formula), with concrete test functions
         f, l, g, W, phi, evaluated at random points to 30 digits.
Check 2: the limits used at the origin (r = 0) and on the axis (sin theta = 0) for regular fields:
         q -> phi_theta / (r cos theta) (axis), phi_r / sin theta (origin), phi_rtheta / cos theta (both);
         phi_theta / r -> phi_rtheta and W / r -> W_r (origin).

Usage: python3 tools/verify_cartesian.py
"""
import random
import sympy as sp

t, x, y, z = sp.symbols('t x y z', real=True)
r_, th_, vp_ = sp.symbols('r theta varphi', positive=True)
w = sp.Rational(4, 5)
mu2, lam = sp.Rational(1), sp.Rational(3, 7)
X = [t, x, y, z]

# concrete test functions of (r, theta): a valid metric (f, l, g > 0), not regular-by-design
s_, c_ = sp.sin(th_), sp.cos(th_)
fF = 1 - sp.Rational(3, 10) * sp.exp(-r_**2 / 2) * (1 + c_**2 / 5)
lF = 1 + sp.Rational(1, 5) * sp.exp(-r_**2 / 3) * (1 + s_**2 / 7)
gF = 1 + sp.Rational(1, 10) * r_**2 * s_**2 * sp.exp(-r_**2) + sp.Rational(1, 20) * c_ * r_ * sp.exp(-r_)
WF = sp.Rational(1, 4) * r_ * sp.exp(-r_**2 / 2) * (1 + c_**2 / 3) + sp.Rational(1, 30) * c_ * r_**2 * sp.exp(-r_)


def phiF(m):
    return (r_ * s_)**m * sp.exp(-r_**2) * (1 + c_**2 / 4 + c_ / 9)


def U(p):
    return mu2 * p**2 + lam * p**4


def check_closed_forms(m, npts=3, tol=1e-22):
    pF = phiF(m)
    rr = sp.sqrt(x**2 + y**2 + z**2)
    sub = {r_: rr, th_: sp.acos(z / rr)}
    f, l, g, W, p = (e.subs(sub) for e in (fF, lF, gF, WF, pF))
    # spherical 4-metric -> Cartesian by the pullback g_cart = J^T g_sph J
    sph = [t, rr, sp.acos(z / rr), sp.atan2(y, x)]
    J = sp.Matrix(4, 4, lambda a, b: sp.diff(sph[a], X[b]))
    rs, ss = rr, sp.sqrt(x**2 + y**2) / rr
    gs = sp.zeros(4, 4)
    gs[0, 0] = -f + l * ss**2 * W**2 / f
    gs[0, 3] = gs[3, 0] = -(l / f) * rs * ss**2 * W
    gs[3, 3] = (l / f) * rs**2 * ss**2
    gs[1, 1] = (l / f) * g
    gs[2, 2] = (l / f) * g * rs**2
    g4 = J.T * gs * J
    # Symbolic: the Cartesian 4-metric and its first partial derivatives. Everything else (inverse,
    # lapse, shift and their derivatives) is computed numerically from these at each point.
    dg4 = [g4.diff(X[k + 1]) for k in range(3)]
    # Phi in Cartesian coordinates, e^{i m varphi} = ((x + i y)/rho)^m
    Phi = p * sp.exp(sp.I * w * t) * ((x + sp.I * y) / sp.sqrt(x**2 + y**2))**m
    dPhi = [sp.diff(Phi, v) for v in X]
    dPhic = [sp.conjugate(d) for d in dPhi]

    random.seed(m)
    worst = 0.0
    for _ in range(npts):
        pt = {x: sp.Rational(random.randint(-150, 150), 100), y: sp.Rational(random.randint(-150, 150), 100),
              z: sp.Rational(random.randint(-150, 150), 100), t: 0}
        N = lambda e: sp.N(e.subs(pt), 40)
        G4 = g4.applyfunc(N)
        G = G4[1:, 1:]
        Ginv = G.inv()
        blo = sp.Matrix([G4[0, i] for i in (1, 2, 3)])        # beta_i = g_ti
        b = Ginv * blo                                        # beta^i = gamma^ij beta_j
        al = sp.sqrt(-G4[0, 0] + (blo.T * b)[0, 0])           # g_tt = -alpha^2 + beta_i beta^i
        dG4 = [M.applyfunc(N) for M in dg4]
        dgam = [[[dG4[k][i + 1, j + 1] for k in range(3)] for j in range(3)] for i in range(3)]
        # d_i beta^k = gamma^kl d_i beta_l - gamma^ka (d_i gamma_ab) beta^b
        dbeta = [[sum(Ginv[k, l_] * dG4[i][0, l_ + 1] for l_ in range(3))
                  - sum(Ginv[k, a] * dG4[i][a + 1, c + 1] * b[c] for a in range(3) for c in range(3))
                  for i in range(3)] for k in range(3)]
        # K_ij = (1/(2 alpha)) (L_beta gamma)_ij with partial derivatives (stationary)
        K4 = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                v = sum(b[k] * dgam[i][j][k] for k in range(3))
                v += sum(G[k, j] * dbeta[k][i] + G[i, k] * dbeta[k][j] for k in range(3))
                K4[i, j] = v / (2 * al)
        # 4D stress tensor and normal
        G4i = G4.inv()
        dP = [N(d) for d in dPhi]
        dPc = [N(d) for d in dPhic]
        kin = sum(G4i[a, c] * dPc[a] * dP[c] for a in range(4) for c in range(4))
        Uv = U(N(p))
        T = sp.Matrix(4, 4, lambda a, c: dPc[a] * dP[c] + dPc[c] * dP[a] - G4[a, c] * (kin + Uv))
        nvec = [1 / al] + [-b[i] / al for i in range(3)]
        rho4 = sum(nvec[a] * nvec[c] * T[a, c] for a in range(4) for c in range(4))
        S4 = [-sum(nvec[c] * T[i + 1, c] for c in range(4)) for i in range(3)]  # n_i = 0
        Pi4 = sum(nvec[a] * dP[a] for a in range(4))
        Phi4 = N(Phi)

        # closed forms
        rv = N(rr)
        thv = sp.acos(N(z) / rv)
        vpv = sp.atan2(N(y), N(x))
        loc = {r_: rv, th_: thv}
        fv, lv, gv, Wv, pv = (sp.N(e.subs(loc), 40) for e in (fF, lF, gF, WF, pF))
        Wr, Wt = sp.N(sp.diff(WF, r_).subs(loc), 40), sp.N(sp.diff(WF, th_).subs(loc), 40)
        pr, pt_ = sp.N(sp.diff(pF, r_).subs(loc), 40), sp.N(sp.diff(pF, th_).subs(loc), 40)
        s, c, cp, sp_ = sp.sin(thv), sp.cos(thv), sp.cos(vpv), sp.sin(vpv)
        n = sp.Matrix([s * cp, s * sp_, c])
        tv = sp.Matrix([c * cp, c * sp_, -s])
        e = sp.Matrix([-sp_, cp, 0])
        A, B = lv * gv / fv, lv / fv
        gam_c = A * sp.eye(3) + (B - A) * e * e.T
        beta_c = -Wv * s * e
        a_ = -(lv * s / (2 * fv**sp.Rational(3, 2))) * (Wr - Wv / rv)
        b_ = -(lv * s / (2 * fv**sp.Rational(3, 2))) * (Wt / rv)
        K_c = a_ * (n * e.T + e * n.T) + b_ * (tv * e.T + e * tv.T)
        q = pv / (rv * s)
        Phi_c = pv * sp.exp(sp.I * m * vpv)
        Pi_c = sp.I * (w + m * Wv / rv) * Phi_c / sp.sqrt(fv)
        dPhi_c = sp.exp(sp.I * m * vpv) * (pr * n + (pt_ / rv) * tv + sp.I * m * q * e)
        rho_c = (w + m * Wv / rv)**2 * pv**2 / fv + (pr**2 + (pt_ / rv)**2) / A + m**2 * q**2 / B + U(pv)
        S_c = -2 * m * (w + m * Wv / rv) * pv * q / sp.sqrt(fv) * e
        gam_inv_c = sp.eye(3) / A + (1 / B - 1 / A) * e * e.T

        D = lambda a, c: abs(sp.N(sp.expand(a - c), 40))
        diffs = {
            'alpha': D(al, sp.sqrt(fv)),
            'beta': max(D(b[i], beta_c[i]) for i in range(3)),
            'gamma': max(D(G[i, j], gam_c[i, j]) for i in range(3) for j in range(3)),
            'gamma_inv': max(D(Ginv[i, j], gam_inv_c[i, j]) for i in range(3) for j in range(3)),
            'K': max(D(K4[i, j], K_c[i, j]) for i in range(3) for j in range(3)),
            'Phi': D(Phi4, Phi_c),
            'dPhi': max(D(dP[i + 1], dPhi_c[i]) for i in range(3)),
            'Pi': D(Pi4, Pi_c),
            'rho': D(rho4, rho_c),
            'S': max(D(S4[i], S_c[i]) for i in range(3)),
        }
        rho4 = sp.N(rho4, 40)
        S4 = [sp.N(v, 40) for v in S4]
        scale = 1 + max(abs(v) for v in list(K4) + [rho4] + S4)
        for k, v in diffs.items():
            worst = max(worst, float(v / scale))
            if v / scale > tol:
                print(f"[FAIL] m={m} {k}: diff {sp.N(v, 5)} at {pt}")
    ok = worst < tol
    print(f"[check] m = {m}: closed forms vs 4D Cartesian computation at {npts} points: "
          f"max rel diff {worst:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_limits():
    """Origin / axis limits for regular fields phi = (r sin)^m h, W = r w(r, theta), h, w smooth."""
    ok = True
    r, th = sp.symbols('r theta', real=True)
    h = 1 + r**2 * sp.cos(th)**2 / 3 + r * sp.cos(th) / 5     # smooth in (x, y, z): powers of r cos = z
    for m in (1, 2, 3):
        p = (r * sp.sin(th))**m * h
        q = p / (r * sp.sin(th))
        pr, pt, prt = sp.diff(p, r), sp.diff(p, th), sp.diff(p, r, th)
        for th0 in (0, sp.pi):          # axis, r > 0
            L = sp.limit(q, th, th0)
            ok &= sp.simplify(L - (pt / (r * sp.cos(th))).subs(th, th0)) == 0 if m == 1 else \
                sp.simplify(L - sp.limit(pt / (r * sp.cos(th)), th, th0)) == 0
        L = sp.limit(q, r, 0)            # origin, theta generic
        ok &= sp.simplify(L - sp.limit(pr / sp.sin(th), r, 0)) == 0
        for th0 in (0, sp.pi):           # origin on the axis
            L = sp.limit(sp.limit(q, r, 0), th, th0)
            ok &= sp.simplify(L - (prt / sp.cos(th)).subs({r: 0, th: th0})) == 0
        ok &= sp.simplify(sp.limit(pt / r, r, 0) - prt.subs(r, 0)) == 0
    Wf = r * (sp.Rational(1, 3) + r**2 * sp.cos(th)**2)  # W/r smooth -> W(0) = 0
    ok &= sp.simplify(sp.limit(Wf / r, r, 0) - sp.diff(Wf, r).subs(r, 0)) == 0
    ok &= sp.simplify(sp.limit(sp.diff(Wf, th) / r, r, 0) - sp.diff(Wf, r, th).subs(r, 0)) == 0
    print(f"[check] origin / axis limits (m = 1, 2, 3): {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    good = all([check_closed_forms(m) for m in (1, 2)]) and check_limits()
    print("[verify_cartesian] all checks passed" if good else "[verify_cartesian] FAILED")
    raise SystemExit(0 if good else 1)

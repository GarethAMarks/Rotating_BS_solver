"""Derive and verify the rotating boson-star field equations; generate C++ residuals.

Metric (Lewis-Papapetrou, Kleihaus et al. 2005 Eq. 11; W is their metric function "omega"):
    ds^2 = -f dt^2 + (l/f) [ g (dr^2 + r^2 dtheta^2) + r^2 sin^2(theta) (dvarphi - W/r dt)^2 ]
Scalar field Phi = phi(r, theta) exp(i (w t + m varphi)), action -dPhi* dPhi - U(phi), G_mn = kappa T_mn.

Step 1: Einstein tensor and stress tensor -> E_mn = G_mn - kappa T_mn; Klein-Gordon equation KG.
Step 2: transcribe Kleihaus et al. App. B.2 (with the Siemonsen & East App. A corrections) and
        check, in exact rational arithmetic, that each equation is a linear combination of the
        E_mn (span test over the second derivatives and the constant term), and that the phi
        equation is a multiple of KG.
Step 3: emit C++ for the verified residuals (and their |term| scales) -> rotating_equations.inc.

Usage: python3 tools/derive_rotating.py [--codegen out.inc]
"""
import sys
import sympy as sp

t, r, th, vp = sp.symbols('t r theta varphi', real=True)
kap, w, m = sp.symbols('kappa omega m', real=True)
X = [t, r, th, vp]
names = ['f', 'l', 'g', 'W', 'p']
F = {n: sp.Function(n)(r, th) for n in names}
f, l, g, W, p = (F[n] for n in names)
s, c = sp.sin(th), sp.cos(th)

# ---------------------------------------------------------------- metric
gmat = sp.zeros(4, 4)
gmat[0, 0] = -f + l * s**2 * W**2 / f
gmat[0, 3] = gmat[3, 0] = -(l / f) * r * s**2 * W
gmat[3, 3] = (l / f) * r**2 * s**2
gmat[1, 1] = (l / f) * g
gmat[2, 2] = (l / f) * g * r**2
ginv = sp.simplify(gmat.inv())


def christoffel():
    G = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for cc in range(b, 4):
                v = 0
                for d in range(4):
                    if ginv[a, d] == 0:
                        continue
                    v += ginv[a, d] * (sp.diff(gmat[d, b], X[cc]) + sp.diff(gmat[d, cc], X[b])
                                       - sp.diff(gmat[b, cc], X[d]))
                G[a][b][cc] = G[a][cc][b] = v / 2
    return G


def ricci(G):
    R = sp.zeros(4, 4)
    for b in range(4):
        for cc in range(b, 4):
            v = 0
            for a in range(4):
                v += sp.diff(G[a][b][cc], X[a]) - sp.diff(G[a][b][a], X[cc])
                for d in range(4):
                    v += G[a][a][d] * G[d][b][cc] - G[a][cc][d] * G[d][b][a]
            R[b, cc] = R[cc, b] = v
    return R


# ---------------------------------------------------------------- symbols for derivatives
sym = {}
repl = []
for n in names:
    fn = F[n]
    for key, expr in [(n + '_rr', sp.diff(fn, r, 2)), (n + '_tt', sp.diff(fn, th, 2)),
                      (n + '_rt', sp.diff(fn, r, th))]:
        sym[key] = sp.Symbol(key, real=True)
        repl.append((expr, sym[key]))
for n in names:
    fn = F[n]
    for key, expr in [(n + '_r', sp.diff(fn, r)), (n + '_t', sp.diff(fn, th))]:
        sym[key] = sp.Symbol(key, real=True)
        repl.append((expr, sym[key]))
for n in names:
    sym[n] = sp.Symbol(n + 's', real=True)
Us, dUs = sp.symbols('U dU', real=True)


def to_symbols(e):
    for a, b in repl:
        e = e.subs(a, b)
    for n in names:
        e = e.subs(F[n], sym[n])
    return e


def derive():
    print("[derive] Christoffels / Ricci ...", flush=True)
    G = christoffel()
    R = ricci(G)
    Rs = sum(ginv[a, b] * R[a, b] for a in range(4) for b in range(4) if ginv[a, b] != 0)
    Gt = R - gmat * Rs / 2
    # T_mn = -g_mn [ (1/2) g^{ab} S_ab + U ] + S_mn,  S_mn = Phi*_m Phi_n + c.c. = 2 (k_m k_n phi^2 + d_m d_n)
    k = [w, 0, 0, m]
    d = [0, sp.diff(p, r), sp.diff(p, th), 0]
    S = sp.Matrix(4, 4, lambda a, b: 2 * (k[a] * k[b] * p**2 + d[a] * d[b]))
    trS = sum(ginv[a, b] * S[a, b] for a in range(4) for b in range(4) if ginv[a, b] != 0)
    T = -gmat * (trS / 2 + Us) + S
    comps = [(0, 0), (0, 3), (3, 3), (1, 1), (1, 2), (2, 2)]
    print("[derive] Einstein components ...", flush=True)
    E = [to_symbols(Gt[a, b] - kap * T[a, b]) for a, b in comps]
    densities(T, gmat, ginv, k)
    # Klein-Gordon: box Phi - dU/d|Phi|^2 Phi = 0, dU/d|Phi|^2 = U'(phi) / (2 phi)
    sqrtg = sp.sqrt(-gmat.det())
    lap = sum(sp.diff(sqrtg * ginv[a, a] * sp.diff(p, X[a]), X[a]) for a in (1, 2)) / sqrtg
    kk = ginv[0, 0] * w**2 + 2 * ginv[0, 3] * w * m + ginv[3, 3] * m**2
    KG = to_symbols(lap - kk * p - dUs / 2)
    return E, KG


DENS = {}


def densities(T, gmat, ginv, k):
    """Volume densities (to integrate over dr dtheta dvarphi), sqrt(-g) included:
       Komar mass  -(2 T^t_t - T^a_a) sqrt(-g)   (Kleihaus Eq. 20 with the sign giving M > 0),
       charge      -j^t sqrt(-g),  j^t = 2 phi^2 (g^tt w + g^tvarphi m)   (Eq. 23),
       ang. mom.   -T^t_varphi sqrt(-g)                                     (Eq. 22)."""
    sqrtg = l**sp.Rational(3, 2) * g * r**2 * s / f
    detchk = sp.simplify(-gmat.det() - sqrtg**2)
    print(f"[dens] sqrt(-g) = l^(3/2) g r^2 sin / f: {detchk == 0}")
    Tm = ginv * T                        # T^a_b
    trT = sum(Tm[a, a] for a in range(4))
    jt = 2 * p**2 * (ginv[0, 0] * w + ginv[0, 3] * m)
    DENS['komar'] = to_symbols(sp.simplify(-(2 * Tm[0, 0] - trT) * sqrtg))
    DENS['charge'] = to_symbols(sp.simplify(-jt * sqrtg))
    DENS['angmom'] = to_symbols(sp.simplify(-Tm[0, 3] * sqrtg))
    print(f"[dens] T^t_varphi = m j^t (=> J = m Q): {sp.simplify(Tm[0, 3] - m * jt) == 0}")
    # flat, m = 0 limit of the Komar density: (4 w^2 phi^2 - 2 U) r^2 sin(theta)
    flat = {sym['f']: 1, sym['l']: 1, sym['g']: 1, sym['W']: 0, m: 0}
    kf = sp.simplify(DENS['komar'].subs(flat) - (4 * w**2 * sym['p']**2 - 2 * Us) * r**2 * s)
    print(f"[dens] flat m=0 Komar density = (4 w^2 phi^2 - 2 U) r^2 sin: {kf == 0}")


# ---------------------------------------------------------------- Kleihaus App. B.2 (+ S&E fixes)
def kleihaus(corrected=True):
    S = sym
    fs, ls, gs, Ws, ps = (S[n] for n in names)
    n_ = m
    sin, cos, cot = s, c, c / s
    pre = -sp.Rational(1, 2) / (fs * ls)
    lhs = {X_: r**2 * S[X_ + '_rr'] + S[X_ + '_tt'] for X_ in names}
    rhs = {}
    # f equation (B6). S&E: "+ f l df_theta" -> "+ f dl_theta df_theta"; U term gains a factor g.
    gU = gs if corrected else 1               # S&E: U term carries a factor g
    lt = S['l_t'] if corrected else ls        # S&E: f l df_t -> f dl_t df_t
    rhs['f'] = pre * (4 * kap * r**2 * fs * ls**2 * gU * Us - 8 * kap * n_**2 * ls**2 * gs * Ws**2 * ps**2
                      - 16 * kap * r * n_ * w * ls**2 * gs * Ws * ps**2 - 8 * kap * r**2 * w**2 * ls**2 * gs * ps**2
                      - 2 * sin**2 * ls**2 * Ws**2 + 4 * r * fs * ls * S['f_r'] + 2 * cot * fs * ls * S['f_t']
                      + fs * lt * S['f_t'] - 2 * ls * (r**2 * S['f_r']**2 + S['f_t']**2)
                      + r**2 * fs * S['l_r'] * S['f_r'] + 4 * r * sin**2 * ls**2 * Ws * S['W_r']
                      - 2 * sin**2 * ls**2 * (r**2 * S['W_r']**2 + S['W_t']**2))
    rhs['g'] = pre * (-8 * kap * n_**2 * fs * ls * gs**2 / sin**2 * ps**2 - 2 * sin**2 * ls**2 / fs * gs * Ws**2
                      + 8 * kap * fs * ls * gs * S['p_t']**2 + 4 * r * fs * ls * S['g_r'] - 2 * cot * fs * ls * S['g_t']
                      - 2 * fs * ls / gs * (r**2 * S['g_r']**2 + S['g_t']**2)
                      + fs * (r**2 * S['l_r'] * S['g_r'] - S['l_t'] * S['g_t'])
                      + 2 * ls * gs / fs * S['f_t']**2 - 2 * cot * fs * gs * S['l_t'] - 3 * fs / ls * gs * S['l_t']**2
                      + 2 * fs * gs * S['l_tt'] + 4 * r * sin**2 * ls**2 * gs / fs * Ws * S['W_r']
                      - 2 * sin**2 * ls**2 * gs / fs * (r**2 * S['W_r']**2 + 2 * S['W_t']**2))
    rhs['l'] = pre * (8 * kap * r**2 * ls**3 * gs * Us + 8 * kap / sin**2 * n_**2 * fs * ls**2 * gs * ps**2
                      - 8 * kap * n_**2 * ls**3 * gs / fs * Ws**2 * ps**2 - 16 * kap * r * n_ * w * ls**3 * gs / fs * Ws * ps**2
                      - 8 * kap * r**2 * w**2 * ls**3 * gs / fs * ps**2 + 6 * r * fs * ls * S['l_r']
                      + 4 * cot * fs * ls * S['l_t'] - fs * (r**2 * S['l_r']**2 + S['l_t']**2))
    rhs['W'] = pre * (-8 * kap / sin**2 * n_**2 * fs * ls * gs * Ws * ps**2 - 8 * kap * r / sin**2 * n_ * w * fs * ls * gs * ps**2
                      - 4 * fs * ls * Ws + 4 * r * fs * ls * S['W_r'] + 6 * cot * fs * ls * S['W_t']
                      - 4 * ls * (r**2 * S['f_r'] * S['W_r'] + S['f_t'] * S['W_t'])
                      + 3 * fs * (r**2 * S['l_r'] * S['W_r'] + S['l_t'] * S['W_t'])
                      + 4 * r * ls * Ws * S['f_r'] - 3 * r * fs * Ws * S['l_r'])
    rhs['p'] = pre * (-r**2 * ls**2 * gs * dUs - 2 / sin**2 * n_**2 * fs * ls * gs * ps
                      + 2 * ls**2 * gs / fs * (Ws * n_ + r * w)**2 * ps + 4 * r * fs * ls * S['p_r']
                      + 2 * cot * fs * ls * S['p_t'] + fs * (r**2 * S['l_r'] * S['p_r'] + S['l_t'] * S['p_t']))
    return {X_: lhs[X_] - rhs[X_] for X_ in names}


# ---------------------------------------------------------------- exact span test
def random_point(seed):
    import random
    rnd = random.Random(seed)
    q = lambda: sp.Rational(rnd.randint(1, 97), rnd.randint(1, 61))
    vals = {r: q(), kap: q(), w: q(), m: sp.Integer(rnd.randint(1, 3)), Us: q(), dUs: q()}
    for n in names:
        vals[sym[n]] = q()
        vals[sym[n + '_r']] = q() - 1
        vals[sym[n + '_t']] = q() - 1
    return vals


def at_point(e, vals):
    # theta with sin = 3/5, cos = 4/5 exactly
    e = e.subs({sp.sin(th): sp.Rational(3, 5), sp.cos(th): sp.Rational(4, 5)})
    return e.subs(vals)


SECOND = [k for k in sym if k.endswith(('_rr', '_tt', '_rt')) and not k.startswith('p')]


def coeff_vector(e, vals):
    """Coefficients of the metric second derivatives and the constant term at a point."""
    e = sp.expand(at_point(e, vals))
    ss = [sym[k] for k in SECOND]
    v = [e.coeff(x) for x in ss]
    const = e.subs({x: 0 for x in ss})
    return sp.Matrix(v + [const])


def verify(E, KG, K):
    ok = True
    for seed in (1, 2, 3):
        vals = random_point(seed)
        # p second derivatives do not enter E; set them for KG via coefficients separately
        Evecs = sp.Matrix.hstack(*[coeff_vector(e, vals) for e in E])
        for X_ in ['f', 'l', 'g', 'W']:
            kv = coeff_vector(K[X_], vals)
            rank_E = Evecs.rank()
            rank_aug = sp.Matrix.hstack(Evecs, kv).rank()
            good = rank_aug == rank_E
            ok &= good
            print(f"  point {seed}: {X_}-equation in span(E_mn): {good}")
        # phi equation vs KG: K_p must equal lambda * KG identically in p_rr, p_tt and lower order
        kp = sp.expand(at_point(K['p'], vals))
        kg = sp.expand(at_point(KG, vals))
        lam = sp.simplify(kp.coeff(sym['p_rr']) / kg.coeff(sym['p_rr']))
        good = sp.simplify(kp - lam * kg) == 0
        ok &= good
        print(f"  point {seed}: phi-equation = {lam} * KG: {good}")
    return ok


def scale_expr(e):
    """Sum of |terms| of an expanded expression (rounding-error scale of the residual).
    Derivative symbols X_r, X_rr, ... are renamed Xa_r, Xa_rr, ...: the C++ side binds them
    to the absolute-value stencils, whose size sets the rounding error of each derivative;
    dU -> dUa likewise."""
    terms = sp.Add.make_args(sp.expand(e))
    ren = {sym[k]: sp.Symbol(k.replace('_', 'a_', 1), real=True)
           for k in sym if '_' in k}
    ren[dUs] = sp.Symbol('dUa', real=True)  # bound to potential_dU_abs_terms (dU can cancel)
    return sum(sp.Abs(tm) for tm in terms).subs(ren)


def check_spherical_reduction(K):
    """m = 0, g = 1, W = 0, theta-independent: Kleihaus B.2 must reduce to the 1D equations
    used in solver.cpp (compute_residuals), divided by r^2."""
    S = sym
    red = {m: 0, S['g']: 1, S['W']: 0}
    for k in ['g_r', 'g_t', 'g_rr', 'g_tt', 'W_r', 'W_t', 'W_rr', 'W_tt',
              'f_t', 'f_tt', 'l_t', 'l_tt', 'p_t', 'p_tt']:
        red[S[k]] = 0
    fs, ls, ps = S['f'], S['l'], S['p']
    f_r, l_r, p_r = S['f_r'], S['l_r'], S['p_r']
    one_d = {
        'f': S['f_rr'] + sp.Rational(1, 2) / (fs * ls) * (4 * kap * fs * ls**2 * Us - 8 * kap * w**2 * ls**2 * ps**2
             + 4 / r * fs * ls * f_r - 2 * ls * f_r**2 + fs * l_r * f_r),
        'l': S['l_rr'] + sp.Rational(1, 2) / (fs * ls) * (8 * kap * ls**3 * Us - 8 * kap * w**2 * ls**3 / fs * ps**2
             + 6 / r * fs * ls * l_r - fs * l_r**2),
        'p': S['p_rr'] + sp.Rational(1, 2) / (fs * ls) * (2 * w**2 * ls**2 / fs * ps - ls**2 * dUs
             + 4 / r * fs * ls * p_r + fs * l_r * p_r),
    }
    ok = True
    for X_, e1 in one_d.items():
        d = sp.simplify(K[X_].subs(red) / r**2 - e1)
        print(f"[reduce] m=0 {X_}-equation equals 1D form in solver.cpp: {d == 0}")
        ok &= (d == 0)
    return ok


def codegen(K, path):
    # C names: fs, ls, gs, Ws, ps, f_r, ..., sin_t, cos_t, kappa, omega, m, U, dU
    rep = {sp.sin(th): sp.Symbol('sin_t'), sp.cos(th): sp.Symbol('cos_t')}
    with open(path, 'w') as out:
        out.write("// Generated by tools/derive_rotating.py -- do not edit.\n")
        out.write("// Kleihaus et al. (2005) App. B.2 with Siemonsen & East (2021) App. A corrections,\n")
        out.write("// verified against the Einstein-Klein-Gordon equations derived from the metric.\n")
        out.write("// Residual R_X = r^2 X_rr + X_tt - rhs_X for X in f, l, g, W, phi (p).\n")
        out.write("// The scales are only needed for the convergence test: guarded by want_scale.\n")
        for X_ in names:
            out.write(f"res_{X_} = {sp.ccode(K[X_].subs(rep))};\n")
        out.write("if (want_scale) {\n")
        for X_ in names:
            out.write(f"scale_{X_} = {sp.ccode(scale_expr(K[X_].subs(rep)))};\n")
        out.write("}\n")
    dpath = path.replace('rotating_equations', 'rotating_densities')
    with open(dpath, 'w') as out:
        out.write("// Generated by tools/derive_rotating.py -- do not edit.\n")
        out.write("// Volume densities over dr dtheta dvarphi (sqrt(-g) included); see densities().\n")
        for key in ['komar', 'charge', 'angmom']:
            out.write(f"dens_{key} = {sp.ccode(DENS[key].subs(rep))};\n")
    print(f"[codegen] wrote {dpath}")
    print(f"[codegen] wrote {path}")


if __name__ == '__main__':
    E, KG = derive()
    K = kleihaus()
    print("[verify] exact span test at 3 rational points (sin = 3/5):", flush=True)
    ok = verify(E, KG, K)
    print("[verify] ALL OK" if ok else "[verify] MISMATCH")
    # Negative control: the f equation as printed in Kleihaus et al. (without the S&E
    # corrections) must fail the same test.
    vals = random_point(1)
    Evecs = sp.Matrix.hstack(*[coeff_vector(e, vals) for e in E])
    kv = coeff_vector(kleihaus(corrected=False)['f'], vals)
    in_span = sp.Matrix.hstack(Evecs, kv).rank() == Evecs.rank()
    print(f"[control] uncorrected Kleihaus f-equation in span(E_mn): {in_span} (expected False)")
    ok &= check_spherical_reduction(K)
    if '--codegen' in sys.argv and ok:
        codegen(K, sys.argv[sys.argv.index('--codegen') + 1])

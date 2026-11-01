"""Dirac stars on the rotating Lewis-Papapetrou metric: tetrad, spin connection, Dirac operator.

Metric as in derive_rotating.py; tetrad (Herdeiro, Perapechka, Radu & Shnir 2019, in our metric
functions): e^0 = sqrt(f) dt, e^1 = sqrt(l g / f) dr, e^2 = sqrt(l g / f) r dtheta,
e^3 = sqrt(l / f) r sin(theta) (dvarphi - W/r dt).
Dirac equation (gamma^a e_a^mu D_mu - mu) Psi = 0, {gamma^a, gamma^b} = 2 eta^{ab}, eta = diag(-1,1,1,1),
D_mu = d_mu + (1/4) omega_mu_ab gamma^a gamma^b, omega_mu^a_b = e^a_nu (d_mu e_b^nu + Gamma^nu_mu,lam e_b^lam).
Ansatz Psi = e^{i (m varphi - w t)} (psi1, psi2, -i psi1*, -i psi2*), psi1 = P + i Q, psi2 = X + i Y.

Step 1 (this file, --rep): find the gamma representation in which the ansatz is consistent, i.e. the
equations for the lower components are fixed multiples of the complex conjugates of those for the
upper ones, so that the 8 real equations reduce to 4.
"""
import sys
import time
import random
import sympy as sp

import derive_rotating as DR

t, r, th, vp = DR.t, DR.r, DR.th, DR.vp
X = DR.X
w, m = DR.w, DR.m
mu = sp.Symbol('mu', positive=True)
f, l, g, W = (DR.F[n] for n in ['f', 'l', 'g', 'W'])
s, c = DR.s, DR.c
I = sp.I


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tetrad():
    """e^a_mu (rows a, cols mu) and inverse e_a^mu."""
    E = sp.zeros(4, 4)
    E[0, 0] = sp.sqrt(f)
    E[1, 1] = sp.sqrt(l * g / f)
    E[2, 2] = sp.sqrt(l * g / f) * r
    E[3, 3] = sp.sqrt(l / f) * r * s
    E[3, 0] = -sp.sqrt(l / f) * r * s * W / r
    Ei = sp.simplify(E.inv())       # Ei[mu, a] = e_a^mu
    return E, Ei


def check_tetrad(E):
    eta = sp.diag(-1, 1, 1, 1)
    gg = sp.simplify(E.T * eta * E - DR.gmat)
    return gg == sp.zeros(4, 4)


def reps():
    s1 = sp.Matrix([[0, 1], [1, 0]])
    s2 = sp.Matrix([[0, -I], [I, 0]])
    s3 = sp.Matrix([[1, 0], [0, -1]])
    Z, E2 = sp.zeros(2), sp.eye(2)
    blk = lambda a, b, c_, d: sp.Matrix(sp.BlockMatrix([[a, b], [c_, d]]))
    out = {}
    # chiral (Weyl), mostly plus: gamma^0 = i [[0,1],[1,0]], gamma^k = i [[0, s_k], [-s_k, 0]]
    out['chiral'] = [I * blk(Z, E2, E2, Z)] + [I * blk(Z, sk, -sk, Z) for sk in (s1, s2, s3)]
    # Dirac: gamma^0 = i diag(1, -1), gamma^k as above
    out['dirac'] = [I * blk(E2, Z, Z, -E2)] + [I * blk(Z, sk, -sk, Z) for sk in (s1, s2, s3)]
    # variants with the spatial matrices without the i (then gamma^0 = i ...) are not Clifford for -+++;
    # sign variants of gamma^0
    out['chiral-'] = [-out['chiral'][0]] + out['chiral'][1:]
    out['dirac-'] = [-out['dirac'][0]] + out['dirac'][1:]
    # The ansatz (psi1, psi2, -i psi1*, -i psi2*) is the fixed set of K u = -i S conj(u), S = block
    # swap. The phase-stripped operator commutes with K iff S conj(gamma^a) S^-1 = -gamma^a for the
    # tetrad legs carrying the phase (a = 0, 3: t, varphi) and +gamma^a for a = 1, 2 (r, theta); the
    # spin connection adds only gamma^1, gamma^2 and the triples [013], [023] for this metric. In
    # the chiral form real sigma_1, sigma_3 give +, imaginary sigma_2 gives -: so sigma_2 goes on e_3.
    out['chiral132'] = [I * blk(Z, E2, E2, Z)] + [I * blk(Z, sk, -sk, Z) for sk in (s1, s3, s2)]
    eta = sp.diag(-1, 1, 1, 1)
    for k_, G in list(out.items()):
        ok = all(sp.simplify(G[a] * G[b] + G[b] * G[a] - 2 * eta[a, b] * sp.eye(4)) == sp.zeros(4)
                 for a in range(4) for b in range(4))
        log(f"representation {k_}: Clifford {ok}")
    return {k_: v for k_, v in out.items() if k_ == 'chiral132' or '--all' in sys.argv}


def dirac_operator(G, E, Ei, Gm):
    """Returns a function Psi(list of 4 expressions, phase stripped) -> 4 equations."""
    eta = sp.diag(-1, 1, 1, 1)
    # omega_mu^a_b = e^a_nu (d_mu e_b^nu + Gamma^nu_{mu lam} e_b^lam)
    om = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    for mu_ in range(4):
        for a in range(4):
            for b in range(4):
                v = 0
                for nu in range(4):
                    if E[a, nu] == 0:
                        continue
                    term = sp.diff(Ei[nu, b], X[mu_]) + sum(Gm[nu][mu_][la] * Ei[la, b] for la in range(4))
                    v += E[a, nu] * term
                om[mu_][a][b] = v
    # omega_mu_ab = eta_ac omega_mu^c_b ; Gamma_mu = (1/4) omega_mu_ab gamma^a gamma^b
    Gam = []
    for mu_ in range(4):
        M_ = sp.zeros(4)
        for a in range(4):
            for b in range(4):
                oab = eta[a, a] * om[mu_][a][b]
                if oab != 0:
                    M_ += oab * G[a] * G[b] / 4
        Gam.append(M_)
    kph = [-I * w, 0, 0, I * m]  # phase e^{i (m varphi - w t)}

    def apply(psi):
        Psi = sp.Matrix(psi)
        out = sp.zeros(4, 1)
        for a in range(4):
            for mu_ in range(4):
                if Ei[mu_, a] == 0:
                    continue
                D = sp.Matrix([sp.diff(Psi[k], X[mu_]) for k in range(4)]) + kph[mu_] * Psi + Gam[mu_] * Psi
                out += Ei[mu_, a] * (G[a] * D)
        return out - mu * Psi
    return apply


def find_representation():
    E, Ei = tetrad()
    log(f"tetrad reproduces the metric: {check_tetrad(E)}")
    log("Christoffels ...")
    Gm = DR.christoffel()
    P, Q, Xf, Y = (sp.Function(n, real=True)(r, th) for n in ('P', 'Q', 'X', 'Y'))
    psi1, psi2 = P + I * Q, Xf + I * Y
    psi = [psi1, psi2, -I * sp.conjugate(psi1), -I * sp.conjugate(psi2)]
    # numeric test point: metric functions as explicit smooth functions, matter as polynomials
    rnd = random.Random(7)
    subsf = {}
    for n, fn in zip(['f', 'l', 'g', 'W'], [f, l, g, W]):
        a0, a1, a2 = (sp.Rational(rnd.randint(1, 9), 10) for _ in range(3))
        subsf[fn] = (1 + a0 * r**2 * sp.cos(th)**2 + a1 * r) if n != 'W' else a2 * r * sp.sin(th)**2
    for fn in (P, Q, Xf, Y):
        a0, a1, a2 = (sp.Rational(rnd.randint(1, 9), 10) for _ in range(3))
        subsf[fn] = a0 * r + a1 * r**2 * sp.cos(th) + a2 * r * sp.sin(th)
    pts = [{r: sp.Rational(7, 5), th: sp.Rational(1, 1), w: sp.Rational(3, 4), m: sp.Rational(1, 2), mu: 1},
           {r: sp.Rational(3, 7), th: sp.Rational(2, 5), w: sp.Rational(1, 3), m: sp.Rational(3, 2), mu: 1}]
    # substitute the explicit test functions first (cheap), then build the operator on them
    E_s = E.subs(subsf).doit()
    Ei_s = Ei.subs(subsf).doit()
    Gm_s = [[[sp.sympify(Gm[a_][b_][c_]).subs(subsf).doit() for c_ in range(4)] for b_ in range(4)] for a_ in range(4)]
    psi_s = [sp.sympify(x).subs(subsf).doit() for x in psi]
    for name, G in reps().items():
        op = dirac_operator(G, E_s, Ei_s, Gm_s)
        Eq = op(psi_s)
        for pt in pts:
            vals = [complex(sp.N(Eq[k].subs(pt), 20)) for k in range(4)]
            r3 = vals[2] / vals[0].conjugate()
            r4 = vals[3] / vals[1].conjugate()
            log(f"{name} at r={pt[r]}, m={pt[m]}: E3/conj(E1) = {r3:.10f}, E4/conj(E2) = {r4:.10f}")
    return
    for name, G in reps().items():
        op = dirac_operator(G, E, Ei, Gm)
        Eq = op(psi)
        vals = [sp.N(sp.simplify(Eq[k].subs(subsf).doit()).subs(pt), 20) for k in range(4)]
        # consistency: E3 = c3 conj(E1), E4 = c4 conj(E2) with the same constant for all points/fields
        rat3 = sp.N(vals[2] / sp.conjugate(vals[0]), 12) if vals[0] != 0 else None
        rat4 = sp.N(vals[3] / sp.conjugate(vals[1]), 12) if vals[1] != 0 else None
        log(f"{name}: E3/conj(E1) = {rat3}, E4/conj(E2) = {rat4}")


if __name__ == '__main__':
    if '--rep' in sys.argv:
        find_representation()


def general_pairing_test():
    """Is there a constant 2x2 A with E_lower = A conj(E_upper) for all field configurations?
    Fit A on two random matter configurations at one metric point, test on a third."""
    E, Ei = tetrad()
    Gm = DR.christoffel()
    rnd = random.Random(11)
    subsf = {}
    for n, fn in zip(['f', 'l', 'g', 'W'], [f, l, g, W]):
        a0, a1, a2 = (sp.Rational(rnd.randint(1, 9), 10) for _ in range(3))
        subsf[fn] = (1 + a0 * r**2 * sp.cos(th)**2 + a1 * r) if n != 'W' else a2 * r * sp.sin(th)**2
    E_s, Ei_s = E.subs(subsf).doit(), Ei.subs(subsf).doit()
    Gm_s = [[[sp.sympify(Gm[a_][b_][c_]).subs(subsf).doit() for c_ in range(4)] for b_ in range(4)] for a_ in range(4)]
    pt = {r: sp.Rational(7, 5), th: sp.Rational(1, 1), w: sp.Rational(3, 4), m: sp.Rational(1, 2), mu: 1}
    import numpy as np
    for name, G in reps().items():
        op = dirac_operator(G, E_s, Ei_s, Gm_s)
        samples = []
        for k_ in range(3):
            fs_ = [sum(sp.Rational(rnd.randint(-9, 9), 10) * mon for mon in (r, r**2 * sp.cos(th), r * sp.sin(th), r**2))
                   for _ in range(4)]
            psi1, psi2 = fs_[0] + I * fs_[1], fs_[2] + I * fs_[3]
            Eq = op([psi1, psi2, -I * sp.conjugate(psi1), -I * sp.conjugate(psi2)])
            samples.append(np.array([complex(sp.N(Eq[q].subs(pt), 18)) for q in range(4)]))
        U = np.array([s_[:2].conj() for s_ in samples[:2]]).T     # columns conj(E_upper)
        L = np.array([s_[2:] for s_ in samples[:2]]).T
        A = L @ np.linalg.inv(U)
        pred = A @ samples[2][:2].conj()
        err = np.abs(pred - samples[2][2:]).max() / np.abs(samples[2][2:]).max()
        log(f"{name}: A = {np.round(A, 6).tolist()}, prediction error on a third configuration {err:.1e}")


if __name__ == '__main__' and '--pairing' in sys.argv:
    general_pairing_test()


# ------------------------------------------------------------------ full derivation + codegen
def derive_full(codegen):
    import derive_matter as DM
    G = reps()['chiral132']
    E, Ei = tetrad()
    log("Christoffels / Einstein tensor ...")
    Gm, Gt = DM.einstein()
    S0 = DM.Symbols(DR.names)
    coef, Gs = DM.metric_coefficients(Gt, S0)
    # Unknowns D1 = (P - X)/sqrt2, D2 = (P + X)/sqrt2, D3 = (Q - Y)/sqrt2, D4 = (Q + Y)/sqrt2: in this
    # representation regularity needs D2 = D4 = 0 on theta = 0 (D1, D3 even there) and D1 = D3 = 0 on
    # theta = pi (D2, D4 even) -- the 1/sin terms of the flat equations; the paper's (P, Q, X, Y) BCs.
    fields = DM.METRIC + ['D1', 'D2', 'D3', 'D4']
    S = DM.Symbols(fields)
    D1, D2, D3, D4 = (S.F[n] for n in ['D1', 'D2', 'D3', 'D4'])
    rt2 = sp.sqrt(2)
    Pf, Xf, Qf, Yf = (D1 + D2) / rt2, (D2 - D1) / rt2, (D3 + D4) / rt2, (D4 - D3) / rt2
    psi1, psi2 = Pf + I * Qf, Xf + I * Yf
    psi = [psi1, psi2, -I * sp.conjugate(psi1), -I * sp.conjugate(psi2)]
    op = dirac_operator(G, E, Ei, Gm)
    log("Dirac equations ...")
    Ev = op(psi)
    # E_lower = -i conj(E_upper) (checked numerically in --pairing): the 4 real equations are
    # Re/Im of the upper two. Multiply by r sqrt(l g / f): the r-derivatives then come as r d_r.
    norm = r * sp.sqrt(l * g / f)
    eqs = {}
    for k_, (re_name, im_name) in enumerate([('E1r', 'E1i'), ('E2r', 'E2i')]):
        e = sp.expand(S.to_symbols(Ev[k_] * norm))
        ec = e.subs(I, -I)
        eqs[re_name] = sp.expand((e + ec) / 2)
        eqs[im_name] = sp.expand((e - ec) / (2 * I))
    # rows: Cauchy-Riemann pairs, signed so that each has + r d_r of its own field
    rows = {'D1': (eqs['E2r'] - eqs['E1r']) / rt2, 'D2': (eqs['E1r'] + eqs['E2r']) / rt2,
            'D3': -(eqs['E2i'] - eqs['E1i']) / rt2, 'D4': -(eqs['E2i'] + eqs['E1i']) / rt2}
    firsts = [S.sym[n + suf] for n in ['D1', 'D2', 'D3', 'D4'] for suf in ('_r', '_t')]
    for k_ in rows:
        rows[k_] = sp.expand(rows[k_])
        present = {str(v): sp.simplify(sp.diff(rows[k_], v)) for v in firsts if rows[k_].has(v)}
        log(f"{k_}-row: first derivatives {present}")
    eqs = rows
    # stress tensor
    log("Dirac stress tensor ...")
    eta = sp.diag(-1, 1, 1, 1)
    Psi = sp.Matrix(psi)
    phase = [-I * w, 0, 0, I * m]
    # spin connection matrices (recomputed as in dirac_operator)
    om = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    for mu_ in range(4):
        for a in range(4):
            for b in range(4):
                v = 0
                for nu in range(4):
                    if E[a, nu] == 0:
                        continue
                    v += E[a, nu] * (sp.diff(Ei[nu, b], X[mu_]) + sum(Gm[nu][mu_][la] * Ei[la, b] for la in range(4)))
                om[mu_][a][b] = v
    Gam = []
    for mu_ in range(4):
        M_ = sp.zeros(4)
        for a in range(4):
            for b in range(4):
                oab = eta[a, a] * om[mu_][a][b]
                if oab != 0:
                    M_ += oab * G[a] * G[b] / 4
        Gam.append(M_)
    D = [sp.Matrix([sp.diff(Psi[q], X[mu_]) for q in range(4)]) + phase[mu_] * Psi + Gam[mu_] * Psi for mu_ in range(4)]
    Psibar = -(Psi.H * G[0])                     # Dirac adjoint (row); Psi^dagger includes conj of the phase-stripped part
    Dbar = [sp.Matrix([[sp.diff(Psibar[q], X[mu_]) for q in range(4)]]) - phase[mu_] * Psibar - Psibar * Gam[mu_]
            for mu_ in range(4)]
    gam_low = [sum((E[a, al] * eta[a, a] * G[a] for a in range(4) if E[a, al] != 0), sp.zeros(4)) for al in range(4)]
    T = sp.zeros(4, 4)
    for al in range(4):
        for be in range(4):
            v = 0
            for (x_, y_) in ((al, be), (be, al)):
                v += (Psibar * gam_low[x_] * D[y_])[0, 0] - (Dbar[x_] * gam_low[y_] * Psi)[0, 0]
            T[al, be] = -I / 2 * v / 2
    Ts = [DM.split_real(S.to_symbols(T[a_, b_]), S.sym, f"T_{a_}{b_}") for a_, b_ in DM.COMPS]
    K0 = DR.kleihaus()
    res = {}
    for Xn in DM.METRIC:
        res[Xn] = sp.expand(K0[Xn].subs(DR.kap, 0) - DR.kap * sum(ci * Ti for ci, Ti in zip(coef[Xn], Ts)))
    log("densities ...")
    ginv = DR.ginv
    Tm = ginv * T
    trT = sum(Tm[q, q] for q in range(4))
    sqrtg = l**sp.Rational(3, 2) * g * r**2 * s / f
    gam_up_t = sum((Ei[0, a] * G[a] for a in range(4) if Ei[0, a] != 0), sp.zeros(4))
    jt = (Psibar * gam_up_t * Psi)[0, 0]
    dens = {'komar': -(2 * Tm[0, 0] - trT) * sqrtg, 'charge': jt * sqrtg, 'angmom': -Tm[0, 3] * sqrtg}
    for k_ in dens:
        dens[k_] = DM.split_real(S.to_symbols(dens[k_]), S.sym, f"density {k_}")
    flat = {S.sym['f']: 1, S.sym['l']: 1, S.sym['g']: 1, S.sym['W']: 0}
    for k_ in dens:
        log(f"flat {k_} density: {sp.factor(sp.expand(dens[k_].subs(flat)))}")
    if codegen:
        res.update(eqs)
        DM.write_inc('dirac', S, res, dens, '..')
    return S, eqs, res, dens


if __name__ == '__main__' and '--full' in sys.argv:
    S, eqs, res, dens = derive_full('--codegen' in sys.argv)
    import pickle
    with open('/tmp/claude-1000/-home-gmarks2000-Documents-codeBlocks-projects-Rotating-BS-solver/4bf67315-abe4-423c-936b-de163d38c60d/dirac_derived.pkl', 'wb') as fh:
        pickle.dump({'eqs': {k: str(v) for k, v in eqs.items()}, 'res': {k: str(v) for k, v in res.items()},
                     'dens': {k: str(v) for k, v in dens.items()}}, fh)
    log("saved derived expressions")

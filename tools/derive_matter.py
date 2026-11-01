"""Derive and verify rotating-star field equations for other matter models; generate C++.

Metric (same as tools/derive_rotating.py; Kleihaus et al. 2005, Lewis-Papapetrou form):
    ds^2 = -f dt^2 + (l/f) [ g (dr^2 + r^2 dtheta^2) + r^2 sin^2(theta) (dvarphi - W/r dt)^2 ]
The four metric equations are the linear combinations sum_ab c^X_ab (G_ab - kappa T_ab) whose
G-part is Kleihaus App. B.2 (with the Siemonsen & East corrections) -- the equations already used
for scalar stars. The coefficients c^X_ab depend on the metric only; they are solved for from the
second-derivative structure of G_ab, and the scalar equations are reproduced exactly (regression)
before the stress tensor is swapped for another matter model.

Models:
  proca   A = e^{i(w t + m varphi)} (i V dt + H1/r dr + H2 dtheta + i H3 sin(theta) dvarphi),
          L = -F Fbar / 4 - mu^2 A Abar / 2 (Herdeiro, Perapechka, Radu & Shnir 2019, with their
          e^{i(m varphi - w t)} conjugated to the e^{i(w t + m varphi)} convention of this code).
          Matter equations: the Proca equation itself, nabla_a F^a_b - mu^2 A_b = 0. Its divergence
          gives mu^2 nabla.A = 0 identically, so the Lorenz condition holds and no gauge modes
          exist. (Adding d_b (nabla.A) would make the system an elliptic vector wave equation, but
          it then also admits A = grad chi with chi a Klein-Gordon bound state of the
          self-gravitating background -- spurious "scalar stars" and singular Jacobians wherever
          the background supports such a state.) The r- and theta-equations are curl-curl like:
          principal parts d_theta^2 H1 - d_r d_theta H2 and d_r^2 H2 - d_r d_theta H1.

Usage: python3 tools/derive_matter.py proca [--codegen]
"""
import sys
import time
import random
import sympy as sp

import derive_rotating as DR  # metric, Christoffels, Ricci, Kleihaus B.2 (scalar), helpers

t, r, th, vp = DR.t, DR.r, DR.th, DR.vp
X = DR.X
kap, w, m = DR.kap, DR.w, DR.m
mu = sp.Symbol('mu', positive=True)
s, c = DR.s, DR.c
gmat, ginv = DR.gmat, DR.ginv
METRIC = ['f', 'l', 'g', 'W']
f, l, g, W = (DR.F[n] for n in METRIC)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ symbols for all fields
class Symbols:
    """Function <-> symbol maps for a field list (derivatives X_r, X_t, X_rr, X_tt, X_rt)."""

    def __init__(self, fields):
        self.fields = fields
        self.F = {n: (DR.F[n] if n in DR.F else sp.Function(n, real=True)(r, th)) for n in fields}
        self.sym, self.repl = {}, []
        for n in fields:
            fn = self.F[n]
            for key, expr in [(n + '_rr', sp.diff(fn, r, 2)), (n + '_tt', sp.diff(fn, th, 2)),
                              (n + '_rt', sp.diff(fn, r, th))]:
                self.sym[key] = sp.Symbol(key, real=True)
                self.repl.append((expr, self.sym[key]))
        for n in fields:
            fn = self.F[n]
            for key, expr in [(n + '_r', sp.diff(fn, r)), (n + '_t', sp.diff(fn, th))]:
                self.sym[key] = sp.Symbol(key, real=True)
                self.repl.append((expr, self.sym[key]))
        for n in fields:
            self.sym[n] = sp.Symbol(n + 's', real=True)

    def to_symbols(self, e):
        for a, b in self.repl:
            e = e.subs(a, b)
        for n in self.fields:
            e = e.subs(self.F[n], self.sym[n])
        return e


# ------------------------------------------------------------ Einstein tensor
def einstein():
    log("Christoffels / Ricci ...")
    Gm = DR.christoffel()
    R = DR.ricci(Gm)
    Rs = sum(ginv[a, b] * R[a, b] for a in range(4) for b in range(4) if ginv[a, b] != 0)
    return Gm, R - gmat * Rs / 2


COMPS = [(0, 0), (0, 3), (3, 3), (1, 1), (1, 2), (2, 2)]


def metric_coefficients(Gt, S):
    """c^X_ab with sum c^X_ab G_ab = Kleihaus B.2 vacuum part (kappa = 0) for X in f, l, g, W."""
    Gs = [sp.expand(S.to_symbols(Gt[a, b])) for a, b in COMPS]
    K0 = DR.kleihaus()
    second = [S.sym[n + suf] for n in METRIC for suf in ('_rr', '_tt', '_rt')]
    cs = sp.symbols('c0:6')
    coef = {}
    for Xn in METRIC:
        k0 = sp.expand(K0[Xn].subs(kap, 0))
        # DR.kleihaus uses derive_rotating's own symbols, identical names -> same Symbol objects
        eqs = []
        for sd in second:
            eqs.append(sp.Eq(sum(ci * sp.diff(Gi, sd) for ci, Gi in zip(cs, Gs)), sp.diff(k0, sd)))
        sol = sp.solve(eqs, cs, dict=True)
        if len(sol) != 1:
            raise RuntimeError(f"coefficients for {Xn}: {sol}")
        cX = [sp.simplify(sol[0].get(ci, 0)) for ci in cs]
        if any(ci in sp.Add(*cX).free_symbols for ci in cs):
            raise RuntimeError(f"coefficients for {Xn} not unique: {cX}")
        coef[Xn] = cX
        log(f"c^{Xn} = {cX}")
    return coef, Gs


def rational_point(syms, seed):
    rnd = random.Random(seed)
    q = lambda: sp.Rational(rnd.randint(1, 97), rnd.randint(1, 61))
    vals = {r: q(), kap: q(), w: q(), m: sp.Integer(rnd.randint(1, 3)), mu: q(),
            DR.Us: q(), DR.dUs: q(), sp.sin(th): sp.Rational(3, 5), sp.cos(th): sp.Rational(4, 5)}
    for k, v in syms.items():
        vals[v] = q() - (1 if '_' in k else 0)
    return vals


def split_real(e, syms, what):
    """Real part of an expression in real symbols; raises if the imaginary part is not zero."""
    e = sp.expand(e)
    if not e.has(sp.I):
        return e
    ec = e.subs(sp.I, -sp.I)
    re_, im_ = sp.expand((e + ec) / 2), sp.expand((e - ec) / (2 * sp.I))
    if not exact_zero(im_, syms):
        raise RuntimeError(f"{what}: imaginary part does not vanish")
    return re_


def exact_zero(e, syms, seeds=(1, 2, 3)):
    for sd in seeds:
        v = rational_point(syms, sd)
        val = sp.nsimplify(e.subs({sp.sin(th): v[sp.sin(th)], sp.cos(th): v[sp.cos(th)]}).subs(v))
        if val != 0:
            return False
    return True


def scalar_regression(coef, Gs):
    """sum c (G - kappa T_scalar) must equal the full scalar Kleihaus equations (with kappa, U)."""
    S = Symbols(DR.names)
    # scalar stress tensor exactly as in derive_rotating.derive()
    p = DR.F['p']
    k = [w, 0, 0, m]
    d = [0, sp.diff(p, r), sp.diff(p, th), 0]
    Smat = sp.Matrix(4, 4, lambda a, b: 2 * (k[a] * k[b] * p**2 + d[a] * d[b]))
    trS = sum(ginv[a, b] * Smat[a, b] for a in range(4) for b in range(4) if ginv[a, b] != 0)
    T = -gmat * (trS / 2 + DR.Us) + Smat
    Ts = [S.to_symbols(T[a, b]) for a, b in COMPS]
    K = DR.kleihaus()
    ok = True
    for Xn in METRIC:
        lhs = sum(ci * (Gi - kap * Ti) for ci, Gi, Ti in zip(coef[Xn], Gs, Ts))
        good = exact_zero(lhs - K[Xn], S.sym)
        log(f"scalar regression: sum c (G - kappa T) == Kleihaus {Xn}-equation: {good}")
        ok &= good
    return ok


# ------------------------------------------------------------ Proca
def proca(coef, Gs, Gm, codegen):
    fields = METRIC + ['V', 'H1', 'H2', 'H3']
    S = Symbols(fields)
    V, H1, H2, H3 = (S.F[n] for n in ['V', 'H1', 'H2', 'H3'])
    I = sp.I
    # A_mu = e^{i chi} a_mu, chi = w t + m varphi; the phase is stripped: d_t -> i w, d_varphi -> i m
    a = [I * V, H1 / r, H2, I * H3 * s]
    k = [I * w, 0, 0, I * m]  # derivative of the phase
    dA = [[sp.diff(a[nu], X[mu_]) + k[mu_] * a[nu] for nu in range(4)] for mu_ in range(4)]  # d_mu A_nu
    Fl = sp.Matrix(4, 4, lambda mu_, nu: dA[mu_][nu] - dA[nu][mu_])          # F_{mu nu} (phase stripped)
    # conjugates: e^{-i chi} abar; d_mu Abar_nu -> conj(a)' - k a* ...
    ab = [sp.conjugate(x) for x in a]
    kb = [-x for x in k]
    dAb = [[sp.diff(ab[nu], X[mu_]) + kb[mu_] * ab[nu] for nu in range(4)] for mu_ in range(4)]
    Fbl = sp.Matrix(4, 4, lambda mu_, nu: dAb[mu_][nu] - dAb[nu][mu_])
    Fu = ginv * Fl * ginv                                                     # F^{mu nu}
    Fbu = ginv * Fbl * ginv
    log("Proca stress tensor ...")
    FF = sum(Fl[i, j] * Fbu[i, j] for i in range(4) for j in range(4))
    Au = ginv * sp.Matrix(a)
    Abu = ginv * sp.Matrix(ab)
    AA = sum(a[i] * Abu[i] for i in range(4))
    T = sp.zeros(4, 4)
    for al in range(4):
        for be in range(4):
            e = 0
            for sg in range(4):
                for ga in range(4):
                    if ginv[sg, ga] != 0:
                        e += (Fl[al, sg] * Fbl[be, ga] + Fbl[al, sg] * Fl[be, ga]) * ginv[sg, ga] / 2
            e += -gmat[al, be] * FF / 4 + mu**2 / 2 * (a[al] * ab[be] + ab[al] * a[be] - gmat[al, be] * AA)
            T[al, be] = e
    Ts = [split_real(S.to_symbols(T[a_, b_]), S.sym, f"T_{a_}{b_}") for a_, b_ in COMPS]
    log("metric equations: sum c (G - kappa T_Proca) ...")
    K0 = DR.kleihaus()
    res = {}
    for Xn in METRIC:
        res[Xn] = sp.expand(K0[Xn].subs(kap, 0) - kap * sum(ci * Ti for ci, Ti in zip(coef[Xn], Ts)))
    # matter equations: nabla_a F^a_b - mu^2 A_b + d_b (nabla . A), phase stripped
    log("Proca matter equations ...")
    sqrtg = l**sp.Rational(3, 2) * g * r**2 * s / f
    divF = [sum(sp.diff(sqrtg * Fu[al, be], X[al]) + k[al] * sqrtg * Fu[al, be] for al in range(4)) / sqrtg
            for be in range(4)]                                              # nabla_a F^{a b}
    divF_low = [sum(gmat[be, ga] * divF[ga] for ga in range(4)) for be in range(4)]
    divA = sum(sp.diff(sqrtg * Au[al], X[al]) + k[al] * sqrtg * Au[al] for al in range(4)) / sqrtg  # nabla . A
    E = [divF_low[be] - mu**2 * a[be] for be in range(4)]
    phase = [I, 1, 1, I * s]                                                  # A_t = i V, ..., A_varphi = i H3 sin
    # principal term normalised to r^2 X_rr (V, H2, H3) or X_tt (H1, which has no H1_rr)
    target = [('V', '_rr', r**2), ('H1', '_tt', 1), ('H2', '_rr', r**2), ('H3', '_rr', r**2)]
    for be, (n, suf, norm) in enumerate(target):
        e = split_real(S.to_symbols(E[be] / phase[be]), S.sym, f"Proca {n}-equation")
        lead = sp.diff(e, S.sym[n + suf])
        e = sp.expand(e * norm / lead)
        second = {kk: sp.simplify(sp.diff(e, S.sym[kk])) for kk in S.sym
                  if kk.endswith(('_rr', '_tt', '_rt')) and e.has(S.sym[kk])}
        log(f"{n}-equation: normalised by {norm} / ({sp.simplify(lead)}); second derivatives: {second}")
        res[n] = e
    # Lorenz condition (diagnostic): nabla . A / (phase of A_t-like i)
    res['lorenz'] = split_real(S.to_symbols(divA), S.sym, "Lorenz condition")  # nabla.A is real here
    # densities: Komar -(2 T^t_t - T) sqrt(-g), charge j^t sqrt(-g), J = -T^t_varphi sqrt(-g)
    log("densities ...")
    Tm = ginv * T
    trT = sum(Tm[i, i] for i in range(4))
    jt = I / 2 * sum(Fbu[0, be] * a[be] - Fu[0, be] * ab[be] for be in range(4))
    # j^t sqrt(-g) is negative for w > 0 in the e^{i(w t + m varphi)} convention (flat limit:
    # -w (H1^2 + H2^2 + H3^2) sin + ...): Q = -int j^t sqrt(-g), as for the scalar (-j^t).
    dens = {'komar': -(2 * Tm[0, 0] - trT) * sqrtg, 'charge': -jt * sqrtg, 'angmom': -Tm[0, 3] * sqrtg}
    for kk in dens:
        dens[kk] = split_real(S.to_symbols(dens[kk]), S.sym, f"density {kk}")
    # flat-space, W = 0 limit of the charge density: its sign fixes the convention Q > 0
    flat = {S.sym['f']: 1, S.sym['l']: 1, S.sym['g']: 1, S.sym['W']: 0}
    qf = sp.factor(sp.expand(dens['charge'].subs(flat)))
    log(f"flat charge density: {qf}")
    lf = sp.factor(sp.expand(res['lorenz'].subs(flat)))
    log(f"flat Lorenz condition nabla.A: {lf}")
    return S, res, dens


def scale_expr(e, S):
    """Sum of |terms|, derivative symbols renamed X_r -> Xa_r (bound to |stencil| values in C++)."""
    terms = sp.Add.make_args(sp.expand(e))
    ren = {S.sym[k]: sp.Symbol(k.replace('_', 'a_', 1), real=True) for k in S.sym if '_' in k}
    return sum(sp.Abs(tm) for tm in terms).subs(ren)


def write_inc(model, S, res, dens, root):
    rep = {sp.sin(th): sp.Symbol('sin_t'), sp.cos(th): sp.Symbol('cos_t')}
    eqs = [n for n in S.fields]
    path = f"{root}/{model}_equations.inc"
    with open(path, 'w') as out:
        out.write("// Generated by tools/derive_matter.py -- do not edit.\n")
        out.write(f"// {model}: metric equations sum c (G - kappa T) (Kleihaus B.2 structure), matter equations\n")
        out.write("// normalised to r^2 X_rr + X_tt + ...; scales guarded by want_scale.\n")
        for n in eqs:
            out.write(f"res_{n} = {sp.ccode(res[n].subs(rep))};\n")
        out.write("if (want_scale) {\n")
        for n in eqs:
            out.write(f"scale_{n} = {sp.ccode(scale_expr(res[n].subs(rep), S))};\n")
        out.write("}\n")
    log(f"wrote {path}")
    path = f"{root}/{model}_densities.inc"
    with open(path, 'w') as out:
        out.write("// Generated by tools/derive_matter.py -- do not edit.\n")
        out.write("// Volume densities over dr dtheta dvarphi (sqrt(-g) included): Komar mass, Noether charge,\n")
        out.write("// angular momentum; and the Lorenz condition nabla.A (diagnostic, should vanish).\n")
        for kk in ['komar', 'charge', 'angmom']:
            out.write(f"dens_{kk} = {sp.ccode(dens[kk].subs(rep))};\n")
        if 'lorenz' in res:
            out.write(f"lorenz = {sp.ccode(res['lorenz'].subs(rep))};\n")
            terms = sp.Add.make_args(sp.expand(res['lorenz'].subs(rep)))
            out.write(f"lorenz_scale = {sp.ccode(sum(sp.Abs(tm) for tm in terms))};\n")
    log(f"wrote {path}")


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'proca'
    Gm, Gt = einstein()
    S0 = Symbols(DR.names)
    coef, Gs = metric_coefficients(Gt, S0)
    ok = scalar_regression(coef, Gs)
    log("scalar regression " + ("OK" if ok else "FAILED"))
    if model == 'proca' and ok:
        S, res, dens = proca(coef, Gs, Gm, '--codegen' in sys.argv)
        if '--codegen' in sys.argv:
            write_inc('proca', S, res, dens, '..')

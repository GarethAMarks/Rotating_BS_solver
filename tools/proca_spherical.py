"""Independent spherical Proca stars: ODEs derived here (Schwarzschild-like gauge, own Einstein
tensor) and solved by shooting -- a check of the conventions of the 2D Proca equations
(tools/derive_matter.py) and a source of fundamental (nodeless) profiles to seed the 2D solver.

Metric ds^2 = -s(R)^2 N(R) dt^2 + dR^2 / N + R^2 dOmega^2, N = 1 - 2 mm(R)/R;
A = e^{-i w t} (a(R) dt + i b(R) dR), L = -F Fbar/4 - mu^2 A Abar / 2, G_mn = 8 pi T_mn (G = 1),
mu = 1. (Brito, Cardoso, Herdeiro & Radu 2016: fundamental static family, M_max ~ 1.058 at
w ~ 0.875.)
ODEs: G^t_t, G^R_R -> mm', s'; Proca R-component (algebraic in b) -> a'; Lorenz condition -> b'.
Shooting: s(0) = 1, a(0) = a0, b(0) = 0, mm(0) = 0; bisect w for the nodeless decaying solution;
physical frequency w / s(inf), mass mm(inf).

Usage: python3 tools/proca_spherical.py --family | --profile a0 out.dat | --seed a0 out.seed
  --seed writes the solution in the isotropic coordinates of the 2D solver (seed_file format:
  r f l g W V H1 H2 H3): ln r = int dR / (R sqrt N) (matched to exterior Schwarzschild at the cut),
  f = s^2 N, l = f (R/r)^2, and, after a constant phase, V = a, H1 = r b dR/dr.
"""
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

R, t, th, ph = sp.symbols('R t theta phi', positive=True)
w = sp.Symbol('omega', positive=True)
mm, s, a, b = (sp.Function(n, real=True)(R) for n in ('mm', 's', 'a', 'b'))
mu = 1


def einstein_mixed(g, X):
    gi = g.inv()
    n = 4
    Gam = [[[sum(gi[a_, d] * (sp.diff(g[d, b_], X[c_]) + sp.diff(g[d, c_], X[b_]) - sp.diff(g[b_, c_], X[d]))
                 for d in range(n)) / 2 for c_ in range(n)] for b_ in range(n)] for a_ in range(n)]
    Ric = sp.zeros(n, n)
    for b_ in range(n):
        for c_ in range(n):
            Ric[b_, c_] = sum(sp.diff(Gam[a_][b_][c_], X[a_]) - sp.diff(Gam[a_][b_][a_], X[c_])
                              + sum(Gam[a_][a_][d] * Gam[d][b_][c_] - Gam[a_][c_][d] * Gam[d][b_][a_] for d in range(n))
                              for a_ in range(n))
    Rs = sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n))
    return sp.simplify(gi * (Ric - g * Rs / 2))


def derive():
    N = 1 - 2 * mm / R
    X = [t, R, th, ph]
    g = sp.diag(-s**2 * N, 1 / N, R**2, R**2 * sp.sin(th)**2)
    gi = g.inv()
    A = [a, sp.I * b, 0, 0]
    kk = [-sp.I * w, 0, 0, 0]                               # phase e^{-i w t} stripped
    dA = [[sp.diff(A[n], X[m_]) + kk[m_] * A[n] for n in range(4)] for m_ in range(4)]
    Fl = sp.Matrix(4, 4, lambda i, j: dA[i][j] - dA[j][i])
    Ab = [a, -sp.I * b, 0, 0]
    dAb = [[sp.diff(Ab[n], X[m_]) - kk[m_] * Ab[n] for n in range(4)] for m_ in range(4)]
    Fbl = sp.Matrix(4, 4, lambda i, j: dAb[i][j] - dAb[j][i])
    Fu, Fbu = gi * Fl * gi, gi * Fbl * gi
    FF = sum(Fl[i, j] * Fbu[i, j] for i in range(4) for j in range(4))
    AA = sum(A[i] * gi[i, j] * Ab[j] for i in range(4) for j in range(4))
    T = sp.zeros(4, 4)
    for al in range(4):
        for be in range(4):
            e = sum((Fl[al, sg] * Fbl[be, ga] + Fbl[al, sg] * Fl[be, ga]) * gi[sg, ga] / 2
                    for sg in range(4) for ga in range(4) if gi[sg, ga] != 0)
            T[al, be] = e - g[al, be] * FF / 4 + sp.Rational(mu**2, 2) * (A[al] * Ab[be] + Ab[al] * A[be] - g[al, be] * AA)
    Tm = sp.simplify(gi * T)
    Gm = einstein_mixed(g, X)
    sqrtg = s * R**2 * sp.sin(th)
    divF = [sum(sp.diff(sqrtg * Fu[al, be], X[al]) + kk[al] * sqrtg * Fu[al, be] for al in range(4)) / sqrtg
            for be in range(4)]
    P = [sp.simplify(divF[be] - mu**2 * sum(gi[be, c] * A[c] for c in range(4))) for be in range(4)]
    divA = sp.simplify(sum(sp.diff(sqrtg * sum(gi[al, c] * A[c] for c in range(4)), X[al])
                           + kk[al] * sqrtg * sum(gi[al, c] * A[c] for c in range(4)) for al in range(4)) / sqrtg)
    d = sp.Derivative
    sol = sp.solve([sp.Eq(Gm[0, 0], 8 * sp.pi * Tm[0, 0]), sp.Eq(Gm[1, 1], 8 * sp.pi * Tm[1, 1])],
                   [d(mm, R), d(s, R)], dict=True)[0]
    ap = sp.solve(P[1], d(a, R))
    bp = sp.solve(divA, d(b, R))
    if len(ap) != 1 or len(bp) != 1:
        raise RuntimeError(f"a' {ap}, b' {bp}")
    # consistency: with a', b', mm', s' substituted, the Proca t-equation must vanish (Lorenz is implied)
    rep = {d(mm, R): sol[d(mm, R)], d(s, R): sol[d(s, R)]}
    apr = ap[0].subs(rep)
    bpr = bp[0].subs(rep).subs(d(a, R), apr)
    Ptr = P[0].subs(d(a, R, 2), sp.diff(apr, R)).subs(d(b, R), bpr)
    Ptr = Ptr.subs(rep).subs(d(a, R), apr).subs(d(b, R), bpr).subs(rep)
    # charge: Q = -int j^t sqrt(-g) for e^{-i w t} -> sign fixed below so that Q > 0
    jt = sp.simplify(sp.I / 2 * sum(Fbu[0, c] * A[c] - Fu[0, c] * Ab[c] for c in range(4)) * s * R**2)
    syms = sp.symbols('mm_ s_ a_ b_')
    to = {mm: syms[0], s: syms[1], a: syms[2], b: syms[3]}
    f_ = lambda e: sp.lambdify((R, *syms, w), sp.simplify(e.subs(rep).subs(d(a, R), apr).subs(d(b, R), bpr).subs(to)), 'numpy')
    chk = sp.lambdify((R, *syms, w), Ptr.subs(to), 'numpy')
    return {"mm": f_(sol[d(mm, R)]), "s": f_(sol[d(s, R)]), "a": f_(apr), "b": f_(bpr), "jt": f_(jt), "Pt": chk,
            "print": {k_: sp.simplify(v) for k_, v in [("mm'", sol[d(mm, R)]), ("s'", sol[d(s, R)]), ("a'", apr), ("b'", bpr), ("jt", jt)]}}


ODE = None


def shoot(a0, wv, Rmax=60.0):
    """Integrate from R = 1e-4; return (event, solution). event: +1 a crossed zero (w too large?),
    -1 a diverged upwards; solution object."""
    def f(Rv, y):
        mm_, s_, a_, b_, q_ = y
        args = (Rv, mm_, s_, a_, b_, wv)
        return [ODE["mm"](*args), ODE["s"](*args), ODE["a"](*args), ODE["b"](*args), ODE["jt"](*args)]
    R0 = 1e-4
    # series: b ~ b1 R from the Lorenz condition at the origin (b' -> b1); use one Euler step from b = 0
    y0 = [0.0, 1.0, a0, 0.0, 0.0]
    dy = f(R0, y0)
    y0 = [y0[i] + R0 * dy[i] if i != 2 else a0 for i in range(5)]
    ev_zero = lambda Rv, y: y[2]
    ev_zero.terminal = True
    ev_blow = lambda Rv, y: abs(y[2]) - 2.0 * abs(a0) - 1e-9
    ev_blow.terminal = True
    sol = solve_ivp(f, (R0, Rmax), y0, rtol=1e-10, atol=1e-13, events=[ev_zero, ev_blow], dense_output=True)
    if sol.t_events[0].size:
        return +1, sol
    if sol.t_events[1].size:
        return -1, sol
    return 0, sol


def fundamental(a0, wlo=0.3, whi=1.6, iters=60):
    """Bisect the trial frequency (s(0) = 1 normalisation) for the nodeless solution."""
    for _ in range(iters):
        wm = 0.5 * (wlo + whi)
        ev, _ = shoot(a0, wm)
        if ev == +1:   # a crosses zero (a node): w too large
            whi = wm
        else:          # a turns up and diverges: w too small
            wlo = wm
    wv = 0.5 * (wlo + whi)
    ev, sol = shoot(a0, wv)
    # stop where the solution starts to deviate: take the point of min |a| before the end
    Rs = sol.t
    k = int(np.argmin(np.abs(sol.y[2]) + 1e-3 * np.abs(sol.y[2][0]) * (Rs < 1.0)))
    mm_inf = sol.y[0][k]
    s_inf = sol.y[1][k]
    Rk = Rs[k]
    # vacuum outside: N -> 1 - 2 M/R, s -> const; frequency measured at infinity
    return dict(a0=a0, w=wv / s_inf, M=mm_inf, Q=abs(sol.y[4][k]) * 4 * np.pi / s_inf, Rcut=Rk, sol=sol, s_inf=s_inf)


def main():
    global ODE
    ODE = derive()
    for k_, v in ODE["print"].items():
        print(f"{k_} = {v}")
    if '--family' in sys.argv:
        print("# a0  omega  M  Q  (fundamental)")
        best = None
        for a0 in np.concatenate([np.linspace(0.005, 0.1, 20), np.linspace(0.11, 0.4, 15)]):
            r_ = fundamental(a0)
            # consistency of the Proca t-equation along the solution
            y = r_["sol"].sol(1.0)
            pt = ODE["Pt"](1.0, *y[:4], r_["w"] * r_["s_inf"])
            print(f"{a0:.4f} {r_['w']:.5f} {r_['M']:.5f} {r_['Q']:.5f}   (cut R = {r_['Rcut']:.1f}, Proca t-eq residual {pt:.1e})", flush=True)
            if best is None or r_["M"] > best["M"]:
                best = r_
        print(f"M_max = {best['M']:.4f} at omega = {best['w']:.4f} (a0 = {best['a0']})")
    if '--profile' in sys.argv:
        a0 = float(sys.argv[sys.argv.index('--profile') + 1])
        out = sys.argv[sys.argv.index('--profile') + 2]
        r_ = fundamental(a0)
        Rk = r_["Rcut"]
        Rs = np.linspace(1e-4, Rk, 4000)
        Y = r_["sol"].sol(Rs)
        np.savetxt(out, np.column_stack([Rs, Y[0], Y[1] / r_["s_inf"], Y[2] / r_["s_inf"], Y[3]]),
                   header=f"omega={r_['w']:.10f} M={r_['M']:.8f} | R mm s a b (s and a rescaled by s_inf)")
        print(f"wrote {out}: omega={r_['w']:.6f} M={r_['M']:.6f}")
    if '--seed' in sys.argv:
        a0 = float(sys.argv[sys.argv.index('--seed') + 1])
        out = sys.argv[sys.argv.index('--seed') + 2]
        r_ = fundamental(a0)
        Rk, M, sinf = r_["Rcut"], r_["M"], r_["s_inf"]
        Rs = np.geomspace(1.2e-4, Rk, 20000)
        Y = r_["sol"].sol(Rs)
        mmR, sR, aR, bR = Y[0], Y[1] / sinf, Y[2] / sinf, Y[3]
        N = 1.0 - 2.0 * mmR / Rs
        # isotropic radius: d ln r / dR = 1 / (R sqrt N), r(Rk) from exterior Schwarzschild;
        # ln r = ln R + ln(rk / Rk) - int_R^Rk (1/sqrt N - 1) dR / R (smooth integrand, N -> 1 at 0)
        rk = 0.5 * (Rk - M + np.sqrt(Rk * Rk - 2.0 * M * Rk))
        smooth = (1.0 / np.sqrt(N) - 1.0) / Rs
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (smooth[1:] + smooth[:-1]) * np.diff(Rs))])
        r = Rs * (rk / Rk) * np.exp(-(cum[-1] - cum))
        integrand = 1.0 / (Rs * np.sqrt(N))
        f = sR**2 * N
        l = f * (Rs / r)**2
        drdR = r * integrand
        H1 = r * bR / drdR
        # beyond the cut: exterior Schwarzschild in isotropic coordinates
        rx = np.geomspace(r[-1] * 1.01, 1e4, 400)
        psi = 1.0 + M / (2.0 * rx)
        fx = ((1.0 - M / (2.0 * rx)) / psi)**2
        lx = fx * psi**4
        z = np.zeros_like(rx)
        data = np.vstack([np.column_stack([r, f, l, np.ones_like(r), np.zeros_like(r), aR, H1, np.zeros_like(r), np.zeros_like(r)]),
                          np.column_stack([rx, fx, lx, np.ones_like(rx), z, z, z, z, z])])
        np.savetxt(out, data, header=f"omega={r_['w']:.12f} M={M:.10f} | r f l g W V H1 H2 H3 (isotropic)")
        print(f"wrote {out}: omega={r_['w']:.10f} M={M:.8f} f(0)={f[0]:.5f}")


if __name__ == '__main__':
    main()

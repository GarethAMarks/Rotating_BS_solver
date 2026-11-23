"""Hamiltonian and momentum constraints of a bs3d_cartesian.dat file (output_cartesian_values = 1).

Uses only the Cartesian data in the file (gamma_ij, K_ij, rho, S_i), nothing of the solver's ansatz:
    H   = R + K^2 - K_ij K^ij - 2 kappa rho
    M_i = D_j K^j_i - D_i K - kappa S_i
and stationarity, which also tests alpha, beta^i and Pi (the constraints do not involve them):
    E_ij = (L_beta gamma)_ij - 2 alpha K_ij      (= d_t gamma_ij, zero)
    P    = alpha Pi + beta^i d_i Phi - i omega Phi  (d_t Phi = i omega Phi)
Partial derivatives d/dx^i by the chain rule from the spherical polar grid:
    d/dr: 4th-order centred differences in the radial index (the solver's grid is uniform in xi),
          divided by dr/di from the same stencil;
    d/dtheta: 4th-order centred differences, continued across the poles through
          (r, -theta, varphi) = (r, theta, varphi + pi) (needs even Nphi);
    d/dvarphi: spectral (FFT).
R_ij needs a second application of these to the Christoffel symbols, so the result is valid for
radial indices 4 .. Nr-5 and away from the poles (the chain rule has 1/sin(theta)): polar index
3 .. Ntheta-4.

Usage: python3 tools/check_cartesian_constraints.py bs3d_cartesian.dat [more files, coarse to fine]
       [--kappa K --omega W] [--nphi N]
kappa and omega come from the bs2d.dat written next to each file (same model) unless given.
A varphi = 0 file (the default output) is first rotated onto N = 16 values of varphi with the rule in
its header (m from that header), so the checks also test that rule.
With several files (solver runs at Nr-1 and Ntheta-1 doubling each time, same radial map and Nphi)
the norms are taken on the points of the coarsest grid, for a convergence order.
"""
import argparse
import os
import re
import sys
import numpy as np

C4 = np.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12])


def model_params(path, kappa=None, omega=None):
    """kappa and omega: from the command line, else from the header of bs2d.dat next to the file
    (the solver writes both for the same model)."""
    if kappa is None or omega is None:
        prof = os.path.join(os.path.dirname(os.path.abspath(path)), 'bs2d.dat')
        if not os.path.exists(prof):
            raise SystemExit(f"{path}: no bs2d.dat next to it; give --kappa and --omega")
        with open(prof) as fh:
            head = fh.readline()
        kappa = float(re.search(r'kappa=(\S+)', head).group(1)) if kappa is None else kappa
        omega = float(re.search(r'omega=(\S+)', head).group(1)) if omega is None else omega
    return kappa, omega


def rotate_slice(D, m, nph):
    """varphi = 0 data (nr, nt, 1, ncol) -> nph values of varphi, by the rule in the file header:
    beta, S -> R v; gamma, K -> R T R^T; (Phi, Pi) -> e^{i m varphi} (Phi, Pi); alpha, rho unchanged."""
    out = np.repeat(D, nph, axis=2)
    comp = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for k in range(nph):
        vp = 2 * np.pi * k / nph
        c, s = np.cos(vp), np.sin(vp)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        S0 = D[:, :, 0]
        O = out[:, :, k]
        O[..., 2] = vp
        for c0 in (3, 7, 27):                      # x y z, beta^i, S_i
            O[..., c0:c0 + 3] = S0[..., c0:c0 + 3] @ R.T
        for c0 in (10, 16):                        # gamma_ij, K_ij
            T = np.zeros(S0.shape[:2] + (3, 3))
            for q, (a, b) in enumerate(comp):
                T[..., a, b] = T[..., b, a] = S0[..., c0 + q]
            T = R @ T @ R.T
            for q, (a, b) in enumerate(comp):
                O[..., c0 + q] = T[..., a, b]
        ph = np.exp(1j * m * vp)
        for c0 in (22, 24):                        # Phi, Pi
            z = (S0[..., c0] + 1j * S0[..., c0 + 1]) * ph
            O[..., c0], O[..., c0 + 1] = z.real, z.imag
    return out


def load(path, kappa=None, omega=None, nphi=16):
    kappa, omega = model_params(path, kappa, omega)
    d = np.loadtxt(path, comments='#')
    if d.shape[1] < 30:
        raise SystemExit(f"{path}: no matter columns (scalar matter only)")
    # grid sizes from the coordinate columns (rows ordered r, theta, varphi, varphi fastest)
    nph = int(np.argmax(d[1:, 2] <= d[0, 2])) + 1 if np.any(d[1:, 2] <= d[0, 2]) else len(d)
    nt = len(np.unique(d[:, 1]))
    nr = len(d) // (nt * nph)
    if nr * nt * nph != len(d):
        raise SystemExit(f"{path}: rows do not form an r x theta x varphi grid")
    D = d.reshape(nr, nt, nph, d.shape[1])
    if nph == 1:
        with open(path) as fh:
            head = ''.join(l for l in fh if l.startswith('#'))
        mm = re.search(r'm = (\S+);', head)
        if not mm:
            raise SystemExit(f"{path}: a single varphi but no rotation rule in the header")
        D = rotate_slice(D, float(mm.group(1)), nphi)
    return kappa, omega, D


def d_index(u, axis):
    """4th-order centred d/d(index) along axis 0 (r) with NaN where the stencil leaves the grid."""
    out = np.full_like(u, np.nan)
    n = u.shape[axis]
    sl = lambda a, b: tuple(slice(a, b) if k == axis else slice(None) for k in range(u.ndim))
    acc = 0
    for o, cw in zip(range(-2, 3), C4):
        if cw != 0.0:
            acc = acc + cw * u[sl(2 + o, n - 2 + o)]
    out[sl(2, n - 2)] = acc
    return out


def d_theta(u, h):
    """4th-order centred d/dtheta (axis 1), continued across the poles via varphi + pi."""
    nt, nph = u.shape[1], u.shape[2]
    half = np.roll(u, nph // 2, axis=2)  # value at varphi + pi
    # extended theta index -2 .. nt+1: theta < 0 and theta > pi from the half-turned copy
    ext = np.concatenate([half[:, 2:0:-1], u, half[:, nt - 2:nt - 4:-1]], axis=1)
    return sum(cw * ext[:, 2 + o:2 + o + nt] for o, cw in zip(range(-2, 3), C4) if cw != 0.0) / h


def d_phi(u):
    nph = u.shape[2]
    k = np.fft.fftfreq(nph, 1.0 / nph)
    if nph % 2 == 0:
        k[nph // 2] = 0.0
    return np.real(np.fft.ifft(1j * k[None, None, :, None] * np.fft.fft(u, axis=2), axis=2)) \
        if u.ndim == 4 else np.real(np.fft.ifft(1j * k[None, None, :] * np.fft.fft(u, axis=2), axis=2))


def constraints(kappa, omega, D):
    r, th, ph = D[:, 0, 0, 0], D[0, :, 0, 1], D[0, 0, :, 2]
    nr, nt, nph = D.shape[:3]
    if nph % 2 or nph < 8:
        raise SystemExit("needs an even Nphi >= 8")
    ht = th[1] - th[0]
    R3 = r[:, None, None] * np.ones((1, nt, nph))
    S, C = np.sin(th)[None, :, None], np.cos(th)[None, :, None]
    Sp, Cp = np.sin(ph)[None, None, :], np.cos(ph)[None, None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        drdi = d_index(r[:, None, None] * np.ones((1, nt, nph)), 0)
        inv_rs = 1.0 / (R3 * S)
        inv_r = 1.0 / R3

    def grad(u):
        """Cartesian partial derivatives of u (shape nr, nt, nph[, k]) -> list of 3 arrays."""
        ex = (lambda a: a[..., None]) if u.ndim == 4 else (lambda a: a)
        ur = d_index(u, 0) / ex(drdi)
        ut = d_theta(u, ht)
        up = d_phi(u)
        with np.errstate(invalid='ignore'):
            dx = ex(S * Cp) * ur + ex(C * Cp * inv_r) * ut - ex(Sp * inv_rs) * up
            dy = ex(S * Sp) * ur + ex(C * Sp * inv_r) * ut + ex(Cp * inv_rs) * up
            dz = ex(C * np.ones_like(Sp)) * ur - ex(S * inv_r) * ut
        return [dx, dy, dz]

    sym = lambda c: np.stack([np.stack([D[..., c + {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
                                         [tuple(sorted((a, b)))] ] for b in range(3)], -1) for a in range(3)], -2)
    g = sym(10)       # gamma_ij, shape (nr, nt, nph, 3, 3)
    K = sym(16)
    rho = D[..., 26]
    Sv = D[..., 27:30]
    gi = np.linalg.inv(g)
    # d_k gamma_ij -> dg[k] shape (..., 3, 3)
    flat = lambda T: T.reshape(T.shape[:3] + (-1,))
    unflat = lambda T, shp: T.reshape(T.shape[:3] + shp)
    dg = [unflat(x_, (3, 3)) for x_ in grad(flat(g))]
    dg = np.stack(dg, axis=-3)  # (..., k, i, j)
    # Gamma^l_ij = 1/2 g^lm (d_i g_mj + d_j g_mi - d_m g_ij)
    Gl = 0.5 * (np.einsum('...imj->...mij', dg) + np.einsum('...jmi->...mij', dg) - dg)  # (..., m, i, j)
    Gam = np.einsum('...lm,...mij->...lij', gi, Gl)
    dGam = np.stack([unflat(x_, (3, 3, 3)) for x_ in grad(flat(Gam))], axis=-4)  # (..., k, l, i, j)
    # R_ij = d_l Gam^l_ij - d_j Gam^l_il + Gam^l_lm Gam^m_ij - Gam^l_jm Gam^m_il
    Ric = (np.einsum('...llij->...ij', dGam) - np.einsum('...jlil->...ij', dGam)
           + np.einsum('...llm,...mij->...ij', Gam, Gam) - np.einsum('...ljm,...mil->...ij', Gam, Gam))
    Rs = np.einsum('...ij,...ij->...', gi, Ric)
    Kup = np.einsum('...ia,...jb,...ab->...ij', gi, gi, K)
    trK = np.einsum('...ij,...ij->...', gi, K)
    KK = np.einsum('...ij,...ij->...', K, Kup)
    H = Rs + trK**2 - KK - 2 * kappa * rho
    Hscale = np.abs(Rs) + trK**2 + np.abs(KK) + 2 * kappa * np.abs(rho)
    # D_k K_ij = d_k K_ij - Gam^l_ki K_lj - Gam^l_kj K_il
    dK = np.stack([unflat(x_, (3, 3)) for x_ in grad(flat(K))], axis=-3)  # (..., k, i, j)
    DK = dK - np.einsum('...lki,...lj->...kij', Gam, K) - np.einsum('...lkj,...il->...kij', Gam, K)
    dtrK = np.stack(grad(trK), axis=-1)
    divK = np.einsum('...jk,...kji->...i', gi, DK)  # g^jk D_k K_ji
    M = divK - dtrK - kappa * Sv
    Mscale = np.abs(divK) + np.abs(dtrK) + kappa * np.abs(Sv)
    # stationarity: (L_beta gamma)_ij = beta^k d_k g_ij + g_kj d_i beta^k + g_ik d_j beta^k
    alpha, beta = D[..., 6], D[..., 7:10]
    dbeta = np.stack(grad(beta), axis=-2)  # (..., i, k) = d_i beta^k
    Lg = (np.einsum('...k,...kij->...ij', beta, dg) + np.einsum('...kj,...ik->...ij', g, dbeta)
          + np.einsum('...ik,...jk->...ij', g, dbeta))
    Est = Lg - 2 * alpha[..., None, None] * K
    Esc = np.abs(Lg) + 2 * np.abs(alpha[..., None, None] * K)
    Phi = D[..., 22] + 1j * D[..., 23]
    Pi = D[..., 24] + 1j * D[..., 25]
    dPhi = np.stack([a + 1j * b for a, b in zip(grad(D[..., 22]), grad(D[..., 23]))], axis=-1)
    bdP = np.einsum('...k,...k->...', beta, dPhi)
    P = alpha * Pi + bdP - 1j * omega * Phi
    Psc = np.abs(alpha * Pi) + np.abs(bdP) + np.abs(omega * Phi)
    valid = np.zeros((nr, nt, nph), bool)
    valid[4:nr - 4, 3:nt - 3, :] = True
    valid &= np.isfinite(H) & np.all(np.isfinite(M), axis=-1)
    return r, th, H, Hscale, M, Mscale, Est, Esc, P, Psc, valid


def norms(r, th, H, Hs, M, Ms, E, Es, P, Ps, mask):
    """L2 norms with the flat volume element r^2 sin(theta), absolute and relative to the L2 norm of
    the sum of the absolute values of the terms; max norms."""
    w = (r[:, None, None]**2 * np.sin(th)[None, :, None]) * np.ones(H.shape)
    l2 = lambda a: np.sqrt(np.sum(w[mask] * a[mask]**2) / np.sum(w[mask]))
    vn = lambda a: np.sqrt(np.sum(np.abs(a)**2, axis=tuple(range(3, a.ndim))))
    out = {}
    for k, a, sc in (('H', H, Hs), ('M', vn(M), vn(Ms)), ('E', vn(E), vn(Es)), ('P', np.abs(P), np.abs(Ps))):
        out[k] = (l2(a), l2(a) / l2(sc), np.max(np.abs(a[mask])))
    return out


def main(paths, kappa=None, omega=None, nphi=16):
    res = []
    for p in paths:
        kap, om, D = load(p, kappa, omega, nphi)
        print(f"{p}: {D.shape[0]} x {D.shape[1]} x {D.shape[2]} points, kappa = {kap:.12g}, omega = {om:.12g}")
        with np.errstate(divide='ignore', invalid='ignore'):  # 1/sin(theta) at the poles (excluded)
            res.append((p, D) + constraints(kap, om, D))
    # common points: those of the first (coarsest) file
    D0 = res[0][1]
    r0, th0 = D0[:, 0, 0, 0], D0[0, :, 0, 1]
    rows = []
    for p, D, r, th, *fields, valid in res:
        ir = np.array([np.argmin(np.abs(r - x_)) for x_ in r0])
        it = np.array([np.argmin(np.abs(th - x_)) for x_ in th0])
        if np.max(np.abs(r[ir] - r0) / (1 + r0)) > 1e-8 or np.max(np.abs(th[it] - th0)) > 1e-10 \
                or D.shape[2] != D0.shape[2]:
            raise SystemExit(f"{p}: its grid does not contain the points of {paths[0]}")
        sub = np.ix_(ir, it, np.arange(D.shape[2]))
        rows.append((p, [f[sub] for f in fields], valid[sub]))
    mask = np.ones(rows[0][2].shape, bool)
    for _, _, valid in rows:
        mask &= valid
    names = {'H': 'Hamiltonian', 'M': 'momentum', 'E': 'd_t gamma_ij', 'P': 'd_t Phi - i omega Phi'}
    print("L2 norm (volume weighted), relative L2 norm (to the |terms|), max norm; order from the L2 norms")
    for key, label in names.items():
        print(f"\n{label}:")
        prev = None
        for p, f, _ in rows:
            n = norms(r0, th0, *f, mask)[key]
            line = f"  {p[-40:]:40s} {n[0]:10.3e} {n[1]:10.3e} {n[2]:10.3e}"
            if prev is not None:
                line += f"   order {np.log2(prev / n[0]):.2f}"
            print(line)
            prev = n[0]
    rr = r0[np.any(mask, axis=(1, 2))]
    print(f"\n(norms over {mask.sum()} common points, r in [{rr.min():.3g}, {rr.max():.3g}], poles excluded)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='bs3d_cartesian.dat files, coarse to fine')
    ap.add_argument('--kappa', type=float, help='gravitational coupling (default: from bs2d.dat next to each file)')
    ap.add_argument('--omega', type=float, help='frequency (default: from bs2d.dat next to each file)')
    ap.add_argument('--nphi', type=int, default=16, help='varphi points for a varphi = 0 file (even, >= 8)')
    a = ap.parse_args()
    main(a.files, a.kappa, a.omega, a.nphi)

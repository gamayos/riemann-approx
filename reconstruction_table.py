"""
Reproduces Table 1: direct zero-reconstruction test (Definition 1) for the folded
budget-N Riemann-Siegel approximant, up to its horizon 2*pi*N^2.

For each budget N:
  - evaluate Z_N on a fine grid up to 2*pi*N^2,
  - take its sign changes as detections,
  - match each true ordinate gamma_k to a detection within half the local spacing,
  - count matched / missed / spurious.

Detection uses sign changes; under simplicity (Condition A) these coincide with the
ordinates. The script is self-contained (mpmath + numpy).
"""
import numpy as np, math
import mpmath as mp
mp.mp.dps = 15
TWO_PI = 2 * math.pi

# true ordinates up to the largest horizon examined (N=8 -> 2 pi 64 ~ 402)
gam = []
k = 1
while True:
    v = float(mp.zetazero(k).imag)
    if v > 420:
        break
    gam.append(v)
    k += 1
gam = np.array(gam)

def local_spacing(g):
    return TWO_PI / math.log(g / TWO_PI) if g > TWO_PI + 0.3 else 6.0

def folded_detections(N, tmax):
    t = np.arange(12.0, tmax, 0.01)
    th = np.array([float(mp.siegeltheta(x)) for x in t])
    nu = np.floor(np.sqrt(t / TWO_PI)).astype(int)
    nmax = int(max(nu.max(), 1))
    logn = np.log(np.arange(1, nmax + 1))
    isq = 1.0 / np.sqrt(np.arange(1, nmax + 1))
    terms = 2 * np.cos(th[None, :] - np.outer(logn, t)) * isq[:, None]
    idx = np.arange(1, nmax + 1)[:, None]
    ZN = np.sum(terms * (idx <= np.minimum(nu, N)[None, :]), axis=0)
    sc = np.where(np.signbit(ZN[:-1]) != np.signbit(ZN[1:]))[0]
    return t[sc] + (ZN[sc] / (ZN[sc] - ZN[sc + 1])) * (t[sc + 1] - t[sc])

print("N   horizon  #zeros  matched  missed  spurious   missed-heights")
for N in [3, 4, 5, 6, 7, 8]:
    hor = TWO_PI * N**2
    det = folded_detections(N, hor)
    zeros = gam[(gam > 14) & (gam <= hor)]
    matched = np.zeros(len(zeros), bool)
    for i, g in enumerate(zeros):
        if len(det) and np.min(np.abs(det - g)) < 0.5 * local_spacing(g):
            matched[i] = True
    spur = 0
    for d in det:
        if d > 14 and len(zeros):
            j = np.argmin(np.abs(zeros - d))
            if np.abs(zeros[j] - d) >= 0.5 * local_spacing(zeros[j]):
                spur += 1
    missed_h = list(np.round(zeros[~matched], 2))
    print("{0}   {1:7.1f}   {2:4d}    {3:4d}    {4:3d}     {5:4d}      {6}".format(
        N, hor, len(zeros), int(matched.sum()), int((~matched).sum()), spur, missed_h))

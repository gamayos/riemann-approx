"""
Validate the revised (three-level) claims numerically BEFORE writing them up.

(1) Mean-square sharpness of the folded horizon:  for t > 2 pi N^2 the omitted
    Riemann-Siegel block should give  <|Z_N - Z|^2> ~ log(t / 2 pi N^2).
(2) DIRECT zero reconstruction (Definition-1 style), not an error knee:
      - list true ordinates gamma_k,
      - detect (folded: sign changes of Z_N; literal: deep local minima of |zeta_N|),
      - match within a local tolerance,
      - count matched / missed / spurious,
      - report the first-failure height.
"""
import numpy as np, math
import mpmath as mp
mp.mp.dps = 15

TWO_PI = 2 * math.pi

# ---- true ordinates up to 600 (comparison only) ----
print("loading true ordinates ...")
gam = []
k = 1
while True:
    v = float(mp.zetazero(k).imag)
    if v > 600: break
    gam.append(v); k += 1
gam = np.array(gam)
print(f"  {len(gam)} ordinates, last = {gam[-1]:.2f}")

def theta(t):  return float(mp.siegeltheta(t))
def Zfun(t):   return float(mp.siegelz(t))

def local_spacing(g):
    return TWO_PI / math.log(g / TWO_PI) if g > TWO_PI + 0.3 else 6.0

# ============================================================
# (1) MEAN-SQUARE SHARPNESS beyond the horizon
# ============================================================
print("\n(1) mean-square sharpness of the folded horizon")
N_ms = 4
hor = TWO_PI * N_ms**2                      # 100.53
tg = np.arange(40.0, 600.0, 0.2)
th = np.array([theta(x) for x in tg])
Zt = np.array([Zfun(x) for x in tg])
nu = np.floor(np.sqrt(tg / TWO_PI)).astype(int)
nmax = int(nu.max())
logn = np.log(np.arange(1, nmax + 1)); isq = 1 / np.sqrt(np.arange(1, nmax + 1))
allterms = 2 * np.cos(th[None, :] - np.outer(logn, tg)) * isq[:, None]
idx = np.arange(1, nmax + 1)[:, None]
ZN = np.sum(allterms * (idx <= np.minimum(nu, N_ms)[None, :]), axis=0)
err2 = (ZN - Zt)**2
# average <err^2> in sliding windows centred at several heights beyond the horizon
print(f"  horizon 2 pi N^2 = {hor:.1f}")
for tc in [150, 200, 300, 400, 500]:
    msk = (tg > tc - 15) & (tg < tc + 15)
    ms = err2[msk].mean()
    pred = math.log(tc / hor)
    print(f"    t~{tc:3d}:  <|Z_N - Z|^2> = {ms:6.3f}   log(t/2piN^2) = {pred:6.3f}   ratio {ms/pred:4.2f}")

# ============================================================
# (2) DIRECT RECONSTRUCTION
# ============================================================
def detections_folded(N, tmax):
    t = np.arange(12.0, tmax, 0.01)
    th = np.array([theta(x) for x in t])
    nu = np.floor(np.sqrt(t / TWO_PI)).astype(int)
    nmax = int(max(nu.max(), 1))
    logn = np.log(np.arange(1, nmax + 1)); isq = 1 / np.sqrt(np.arange(1, nmax + 1))
    terms = 2 * np.cos(th[None, :] - np.outer(logn, t)) * isq[:, None]
    idx = np.arange(1, nmax + 1)[:, None]
    ZN = np.sum(terms * (idx <= np.minimum(nu, N)[None, :]), axis=0)
    sc = np.where(np.signbit(ZN[:-1]) != np.signbit(ZN[1:]))[0]
    det = t[sc] + (ZN[sc] / (ZN[sc] - ZN[sc + 1])) * (t[sc + 1] - t[sc])
    return det

def detections_literal(N, tmax, depth=0.45):
    t = np.arange(max(2.0, math.sqrt(N)), tmax, 0.01)
    n = np.arange(1, N + 1)
    mod = np.abs((n[:, None] ** (-0.5 - 1j * t[None, :])).sum(axis=0))
    # running median (sliding window ~ 6 in t -> 600 samples)
    w = 601
    pad = np.pad(mod, (w // 2, w // 2), mode="edge")
    runmed = np.array([np.median(pad[i:i + w]) for i in range(0, len(mod), 10)])
    runmed = np.interp(np.arange(len(mod)), np.arange(0, len(mod), 10), runmed)
    loc = np.where((mod[1:-1] < mod[:-2]) & (mod[1:-1] < mod[2:]))[0] + 1
    deep = loc[mod[loc] < depth * runmed[loc]]
    return t[deep]

def score(det, tmax):
    zeros = gam[gam <= tmax]
    matched = np.zeros(len(zeros), bool)
    used = np.zeros(len(det), bool)
    for i, g in enumerate(zeros):
        d = 0.5 * local_spacing(g)
        if len(det):
            j = np.argmin(np.abs(det - g))
            if abs(det[j] - g) < d:
                matched[i] = True; used[j] = True
    # spurious: a detection with no zero within that zero's tolerance
    spurious = []
    for j, dd in enumerate(det):
        near = np.abs(zeros - dd)
        if len(zeros):
            i = np.argmin(near)
            if near[i] >= 0.5 * local_spacing(zeros[i]):
                spurious.append(dd)
    spurious = np.array(spurious)
    # first-failure height: first missed zero OR first spurious detection (above warm-up 14)
    miss_pos = zeros[(~matched) & (zeros > 14)]
    spur_pos = spurious[spurious > 14]
    cand = []
    if len(miss_pos): cand.append(miss_pos.min())
    if len(spur_pos): cand.append(spur_pos.min())
    ff = min(cand) if cand else tmax
    return zeros, matched, spurious, ff

print("\n(2a) FOLDED direct reconstruction")
print("  N   horizon   first-fail   matched/zeros<=ff   spurious<=ff")
fold_N, fold_ff = [], []
for N in [2, 3, 4, 5, 6, 7, 8]:
    hor = TWO_PI * N**2
    det = detections_folded(N, 1.45 * hor)
    zeros, matched, spurious, ff = score(det, 1.45 * hor)
    below = zeros <= ff
    sp_below = (spurious <= ff).sum() if len(spurious) else 0
    fold_N.append(N); fold_ff.append(ff)
    print(f"  {N}   {hor:7.1f}   {ff:8.1f}     {matched[below].sum():3d}/{below.sum():<3d}            {sp_below}")

print("\n(2b) LITERAL direct reconstruction")
print("  N   2piN     first-fail   matched/zeros<=ff   spurious<=ff")
lit_N, lit_ff = [], []
for N in [3, 4, 5, 6, 7, 8, 9, 12]:
    lin = TWO_PI * N
    det = detections_literal(N, 1.7 * lin)
    zeros, matched, spurious, ff = score(det, 1.7 * lin)
    below = zeros <= ff
    sp_below = (spurious <= ff).sum() if len(spurious) else 0
    lit_N.append(N); lit_ff.append(ff)
    print(f"  {N}   {lin:6.1f}   {ff:8.1f}     {matched[below].sum():3d}/{below.sum():<3d}            {sp_below}")

# fits
fold_N = np.array(fold_N); fold_ff = np.array(fold_ff)
lit_N = np.array(lit_N); lit_ff = np.array(lit_ff)
ef = np.polyfit(np.log(fold_N), np.log(fold_ff), 1)[0]
sl = np.polyfit(lit_N, lit_ff, 1)[0]
print(f"\nfolded first-failure exponent  = {ef:.2f}  (expect 2)")
print(f"literal first-failure slope    = {sl:.2f}  per integer (2pi={TWO_PI:.2f})")
np.savez("recon.npz", fold_N=fold_N, fold_ff=fold_ff, lit_N=lit_N, lit_ff=lit_ff)
print("saved recon.npz")

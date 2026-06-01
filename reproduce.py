"""
Regenerate the four manuscript figures as clean vector PDFs (no editorial titles;
descriptions live in the LaTeX captions).  All quantities recomputed from scratch.

  fig_folded_zeros.pdf : Riemann-Siegel main sum Z_{nu(t)}(t) vanishing at every gamma
  fig_naive_band.pdf   : |zeta_N(1/2+it)| faithful only while t <~ 2 pi N
  fig_reach.pdf        : log-log reach law -- naive 2 pi N vs folded 2 pi N^2
  fig_gauss.pdf        : |g(chi_k)| = sqrt(p) for every nontrivial character (self-duality avatar)
"""
import numpy as np, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
from sympy import isprime, factorint
mp.mp.dps = 15

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "axes.titlesize": 10.5,
    "figure.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ----------------------------------------------------------------------------
# Shared zeta machinery
# ----------------------------------------------------------------------------
print("computing true zeros up to 205 ...")
gammas = []
k = 1
while True:
    v = float(mp.zetazero(k).imag)
    if v > 205: break
    gammas.append(v); k += 1
gammas = np.array(gammas)
print(f"  {len(gammas)} zeros, last = {gammas[-1]:.2f}")

t = np.arange(2.0, 680.0, 0.2)
print(f"computing true Z(t), theta(t) on {len(t)} points ...")
Ztrue = np.array([float(mp.siegelz(x)) for x in t])
theta = np.array([float(mp.siegeltheta(x)) for x in t])
zeta_true = Ztrue * np.exp(-1j * theta)                # zeta(1/2+it) = e^{-i theta} Z
nu = np.floor(np.sqrt(t / (2 * math.pi))).astype(int)
nmax = int(nu.max())
logn = np.log(np.arange(1, nmax + 1)); isq = 1.0 / np.sqrt(np.arange(1, nmax + 1))
terms = 2 * np.cos(theta[None, :] - np.outer(logn, t)) * isq[:, None]

def Zmain(mcount):
    idx = np.arange(1, nmax + 1)[:, None]
    return np.sum(terms * (idx <= mcount[None, :]), axis=0)

def partial(N):
    n = np.arange(1, N + 1)
    return (n[:, None] ** (-0.5 - 1j * t[None, :])).sum(axis=0)

def smooth(a, w=15): return np.convolve(a, np.ones(w) / w, mode="same")
def knee(err, thr=0.8):
    bad = smooth(err) > thr
    sus = np.convolve(bad, np.ones(10), mode="same") >= 8
    i = np.flatnonzero(sus)
    return t[i[0]] if len(i) else np.nan

# reach measurements
B_folded = np.array([2, 3, 4, 5, 6, 7, 8])
R_folded = np.array([knee(np.abs(Zmain(np.minimum(nu, N)) - Ztrue)) for N in B_folded])
exp_folded, _ = np.polyfit(np.log(B_folded + 1), np.log(R_folded), 1)

B_naive = np.array([3, 4, 5, 6, 7, 8, 9])
R_naive = np.array([knee(np.abs(partial(N) - zeta_true)) for N in B_naive])
slope_naive = np.polyfit(B_naive, R_naive, 1)[0]
print(f"folded exponent (vs term count) = {exp_folded:.2f};  naive slope = {slope_naive:.2f} per integer (2pi={2*math.pi:.2f})")

# ----------------------------------------------------------------------------
# fig_folded_zeros.pdf
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.7))
sel = (t >= 8) & (t < 200); Zp = Zmain(nu)
ax.plot(t[sel], Zp[sel], color="#1b2631", lw=0.9)
ax.axhline(0, color="0.55", lw=0.7)
for g in gammas[gammas < 200]:
    ax.axvline(g, color="#b03a2e", lw=0.7, ls=(0, (4, 3)), alpha=0.65)
ax.set_xlim(8, 200); ax.set_ylim(-4.2, 4.2)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$Z_{\nu(t)}(t)$")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_folded_zeros.pdf", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------
# fig_naive_band.pdf
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.9))
m = (t >= 2) & (t <= 100)
ax.plot(t[m], np.abs(Ztrue)[m], color="#117a65", lw=1.5, label=r"$|\zeta(\frac{1}{2}+it)|$")
ax.plot(t[m], np.abs(partial(5))[m], color="#2e6fb0", lw=0.95, label=r"$|\zeta_{5}(\frac{1}{2}+it)|$")
ax.plot(t[m], np.abs(partial(12))[m], color="#b03a2e", lw=0.95, alpha=0.85, label=r"$|\zeta_{12}(\frac{1}{2}+it)|$")
for Nval, col in [(5, "#2e6fb0"), (12, "#b03a2e")]:
    ax.axvline(2 * math.pi * Nval, color=col, lw=1.0, ls=":", alpha=0.9)
    ax.text(2 * math.pi * Nval, 3.75, rf"$2\pi\!\cdot\!{Nval}$", color=col, ha="center", fontsize=8.5)
ax.set_xlim(2, 100); ax.set_ylim(0, 4)
ax.set_xlabel(r"$t$"); ax.set_ylabel("modulus")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95, ncol=3, handlelength=1.4, columnspacing=1.0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_naive_band.pdf", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------
# fig_reach.pdf  (hero)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
xx = np.linspace(1.9, 9.4, 100)
ax.plot(xx, 2 * math.pi * xx, color="#2e6fb0", lw=1.5, label=r"$2\pi N$ (naive, linear)")
ax.plot(xx, 2 * math.pi * xx**2, color="#b03a2e", lw=1.6, label=r"$2\pi N^{2}$ (folded, quadratic)")
ax.plot(B_naive, R_naive, "s", color="#1b3a5b", ms=7, zorder=5,
        label=rf"naive, measured (slope $\approx{slope_naive:.1f}$)")
ax.plot(B_folded, R_folded, "o", color="#1b2631", ms=7, zorder=5,
        label=rf"folded, measured (exp. $\approx{exp_folded:.2f}$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"term budget $N$ (number of integers)")
ax.set_ylabel(r"approximation horizon height")
ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9]); ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9])
ax.legend(loc="upper left", fontsize=8.6, framealpha=0.96)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_reach.pdf", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------
# fig_gauss.pdf
# ----------------------------------------------------------------------------
def sc_prime_below(N):
    n = N - 1
    while not (n % 4 == 1 and isprime(n)): n -= 1
    return n
p = sc_prime_below(100000); n = p - 1
fac = list(factorint(p - 1).keys())
g = next(gg for gg in range(2, p) if all(pow(gg, (p - 1) // q, p) != 1 for q in fac))
pw = np.empty(n, dtype=np.int64); cur = 1
for j in range(n):
    pw[j] = cur; cur = (cur * g) % p
v = np.exp(2j * np.pi * pw / p)
G = np.abs(np.fft.fft(v))
sqrtp = math.sqrt(p)
maxdev = np.max(np.abs(G[1:] - sqrtp))
print(f"Gauss: p={p}, max||g|-sqrt(p)| = {maxdev:.2e}")

fig, ax = plt.subplots(figsize=(7.2, 2.7))
ks = np.arange(n)
ax.plot(ks[1::25], G[1::25], ".", ms=1.6, color="#1f6f8b", alpha=0.5)  # subsample for vector size
ax.axhline(sqrtp, color="#b03a2e", lw=1.4, label=rf"$\sqrt{{p}}\approx{sqrtp:.2f}$")
ax.plot(0, G[0], "o", color="#e08214", ms=6, label=r"trivial $\chi_0:\ |g|=1$")
ax.set_xlim(-1500, n + 1500); ax.set_ylim(0, sqrtp * 1.28)
ax.set_xlabel(r"character index $k$"); ax.set_ylabel(r"$|g(\chi_k)|$")
ax.legend(loc="center right", fontsize=9, framealpha=0.95)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gauss.pdf", bbox_inches="tight"); plt.close(fig)

# stash numbers the manuscript quotes
with open("_numbers.txt", "w") as f:
    f.write(f"p={p}\nsqrtp={sqrtp:.6f}\ngauss_maxdev={maxdev:.2e}\n")
    f.write(f"folded_exponent={exp_folded:.3f}\nnaive_slope={slope_naive:.3f}\n")
    f.write("R_folded=" + ",".join(f"{N}:{r:.1f}" for N, r in zip(B_folded, R_folded)) + "\n")
    f.write("R_naive=" + ",".join(f"{N}:{r:.1f}" for N, r in zip(B_naive, R_naive)) + "\n")
print("figures written to", OUT)
print(open("_numbers.txt").read())

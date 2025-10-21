"""
Shortcut Distillation (B40DI) — NIST Antoine (bar, K)
Eigenvalue Underwood Implementation
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# NIST Antoine data (bar, K)
# log10(P_bar) = A - B / (T + C)
# Each component has one or more valid temperature ranges from NIST.
# The correct range must be chosen based on the current temperature.
# -----------------------------------------------------------------------------
NIST_ANTOINE_RANGES: Dict[str, List[Tuple[float, float, float, float, float, str]]] = {
    "n-butane": [
        (135.42, 212.89, 4.70812, 1200.475, -13.013, "Carruth & Kobayashi (1973)"),
        (195.11, 272.81, 3.85002, 909.65,   -36.146, "Aston & Messerly (1940)"),
        (272.66, 425.12, 4.35576, 1175.581,  -2.071, "Das, Reed, et al. (1973)"),
    ],
    "1-butene": [
        (195.70, 269.40, 4.24696, 1099.207, -8.256, "Coffin & Maass (1928)"),
    ],
    "i-butane": [
        (188.06, 261.54, 3.94417, 912.141,  -29.808, "Aston, Kennedy, et al. (1940)"),
        (261.31, 408.12, 4.32810, 1132.108,   0.918, "Das, Reed, et al. (1973)"),
    ],
    "cis-2-butene": [
        (203.06, 295.91, 3.98744, 957.06, -36.504, "Scott, Ferguson, et al. (1944)"),
    ],
    "trans-2-butene": [
        (201.70, 274.13, 4.04360, 982.166, -30.775, "Guttman & Pitzer (1945)"),
    ],
}

# -----------------------------------------------------------------------------
# Antoine + K‑value functions
# -----------------------------------------------------------------------------
def _pick_antoine_set(component: str, T_K: float):
    """Selects the correct Antoine parameter set for the current temperature.
    Falls back to the nearest valid range midpoint if T lies outside all ranges."""
    sets = NIST_ANTOINE_RANGES[component]
    for (tmin, tmax, A, B, C, ref) in sets:
        if tmin <= T_K <= tmax:
            return (A, B, C, ref, True)
    mids = [0.5 * (tmin + tmax) for (tmin, tmax, *_rest) in sets]
    idx = int(np.argmin([abs(T_K - m) for m in mids]))
    A, B, C, ref = sets[idx][2:6]
    print(f"⚠️  {component}: T={T_K:.2f} K outside all NIST ranges — extrapolating ({ref}).")
    return (A, B, C, ref, False)

def antoine_psat_bar(component: str, T_K: float) -> float:
    """Returns Psat [bar] using the appropriate NIST Antoine constants."""
    A, B, C, ref, ok = _pick_antoine_set(component, T_K)
    return 10 ** (A - B / (T_K + C))

def K_value(component: str, T_K: float, P_bar: float) -> float:
    """Equilibrium constant K = Psat / P."""
    return antoine_psat_bar(component, T_K) / P_bar

# -----------------------------------------------------------------------------
# Bubble point temperature solver
# Iterates until Σx_i*K_i = 1 (within 1e‑9 tolerance)
# -----------------------------------------------------------------------------
def bubble_point_TK(components: List[str], x: np.ndarray, P_bar: float, T_guess=300.0):
    T = float(T_guess)
    for _ in range(120):
        K = np.array([K_value(c, T, P_bar) for c in components])
        f = np.dot(x, K) - 1.0
        if abs(f) < 1e-9:
            return T
        dT = 0.5
        Kp = np.array([K_value(c, T + dT, P_bar) for c in components])
        fp = np.dot(x, Kp) - 1.0
        dfdT = (fp - f) / dT
        T -= f / dfdT if abs(dfdT) > 1e-12 else math.copysign(0.5, f)
    return T

# -----------------------------------------------------------------------------
# Average relative volatility (ᾱ) w.r.t. the heavy key
# Computed as geometric mean of α_top and α_bottom values.
# -----------------------------------------------------------------------------
def average_alpha_wrt_HK(components: List[str], HK: str, T_top_K: float, T_bottom_K: float, P_bar: float):
    K_top = np.array([K_value(c, T_top_K, P_bar) for c in components])
    K_bot = np.array([K_value(c, T_bottom_K, P_bar) for c in components])
    iHK = components.index(HK)
    return {c: max(1e-12, np.sqrt(K_top[i] / K_top[iHK] * K_bot[i] / K_bot[iHK])) for i, c in enumerate(components)}

# -----------------------------------------------------------------------------
# Dataclass for design specifications
# -----------------------------------------------------------------------------
@dataclass
class Specs:
    F_kmol_h: float
    P_bar: float
    q: float
    condenser_type: str
    reboiler_type: str
    LK: str
    HK: str
    recov_HK_to_B: float
    max_LK_in_B_molfrac: float
    RR_factor: float
    R_override: Optional[float] = None

# -----------------------------------------------------------------------------
# Product split heuristic
# Distributes feed into distillate and bottoms based on key component behaviour.
# -----------------------------------------------------------------------------
def compute_product_splits(components, zF, s: Specs):
    idx = {c: i for i, c in enumerate(components)}
    iLK, iHK = idx[s.LK], idx[s.HK]
    F = s.F_kmol_h
    d = np.zeros_like(zF)
    b = np.zeros_like(zF)

    # Basic key recovery assumptions
    F_HK = F * zF[iHK]
    b[iHK] = s.recov_HK_to_B * F_HK
    d[iHK] = F_HK - b[iHK]

    F_LK = F * zF[iLK]
    d[iLK] = 0.98 * F_LK
    b[iLK] = F_LK - d[iLK]

    # Middle components split based on relative volatility
    T_mid = 310.0  # nominal temperature
    K_mid = np.array([K_value(c, T_mid, s.P_bar) for c in components])
    for j in range(len(components)):
        if j in (iLK, iHK):
            continue
        if K_mid[j] >= K_mid[iLK]:
            d[j] = F * zF[j]
        elif K_mid[j] <= K_mid[iHK]:
            b[j] = F * zF[j]
        else:
            d[j] = 0.5 * F * zF[j]
            b[j] = 0.5 * F * zF[j]

    # Total flows
    D = np.sum(d)
    B = np.sum(b)
    return d / D, b / B, D, B

# -----------------------------------------------------------------------------
# Shortcut Distillation Core Equations
# -----------------------------------------------------------------------------

# Minimum stages from Fenske equation
def fenske_Nmin(xD_LK, xD_HK, xB_LK, xB_HK, alpha):
    """Minimum number of stages (total reflux) from Fenske equation."""
    return math.log((xD_LK / xD_HK) * (xB_HK / xB_LK)) / math.log(alpha)

# Underwood polynomial (see paper for derivation)
def underwood_poly(alphas, zF, q):
    """Constructs the Underwood polynomial (Uses polynomial eigenvalue method)."""
    n = len(alphas)
    p = np.poly(alphas)
    s = np.zeros_like(p)
    for i in range(n):
        Qi, _ = np.polydiv(p, [1, -alphas[i]])
        Qi = np.concatenate(([0.0], Qi))
        s += zF[i] * Qi
    return np.poly1d(s - (1 - q) * p)

# Solve for Underwood θ using eigenvalue method
def solve_theta_eigen(alphas, zF, q, iLK, iHK):
    """Solves for Underwood θ using eigenvalue polynomial roots."""
    poly = underwood_poly(alphas, zF, q)
    roots = np.roots(poly)
    reals = roots[np.isreal(roots)].real
    lo, hi = alphas[iHK] + 1e-9, alphas[iLK] - 1e-9
    candidates = [r for r in reals if lo < r < hi]
    if not candidates:
        raise ValueError("No θ root between α_HK and α_LK")
    return min(candidates, key=lambda r: min(abs(r - a) for a in alphas))

# Minimum reflux ratio from Underwood
def underwood_Rmin(alphas, xD, theta):
    """Minimum reflux ratio (Underwood)."""
    return sum((a * x) / (a - theta + 1e-14) for a, x in zip(alphas, xD)) - 1.0

# Number of stages at finite reflux from Gilliland correlation
def gilliland_N(Nmin, R, Rmin):
    """Number of stages at finite reflux (Eduljee correlation form)."""
    X = (R - Rmin) / (R + 1.0)
    Y = 1.0 - math.exp((1 + 54.4 * X) * (X - 1.0) / 11.0)
    return (Y + Nmin) / (1 - Y)

# Feed stage location from Kirkbride equation
def kirkbride_ratio(B, D, zF_LK, zF_HK, xB_LK, xD_HK):
    """Feed stage location from Kirkbride equation."""
    eps = 1e-16
    term = (B / max(D, eps)) * ((zF_HK + eps) / (zF_LK + eps)) * ((xB_LK + eps) / (xD_HK + eps)) ** 2
    return term ** 0.206

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    components = ["i-butane", "n-butane", "1-butene", "trans-2-butene", "cis-2-butene"]
    zF = np.array([0.329, 0.172, 0.264, 0.156, 0.079])
    specs = Specs(
        F_kmol_h=1000.0, P_bar=1.0, q=0.2,
        condenser_type="total", reboiler_type="partial",
        LK="1-butene", HK="n-butane",
        recov_HK_to_B=0.95, max_LK_in_B_molfrac=0.02,
        RR_factor=1.2, R_override=None
    )

    # --- Product splits
    xD, xB, D, B = compute_product_splits(components, zF, specs)

    # --- End temperatures (bubble points)
    T_top = bubble_point_TK(components, xD, specs.P_bar)
    T_bottom = bubble_point_TK(components, xB, specs.P_bar)

    # --- α averages
    alpha_dict = average_alpha_wrt_HK(components, specs.HK, T_top, T_bottom, specs.P_bar)
    alpha_list = [alpha_dict[c] for c in components]
    iLK = components.index(specs.LK)
    iHK = components.index(specs.HK)

    # --- Fenske
    alpha_LK_HK = alpha_dict[specs.LK] / alpha_dict[specs.HK]
    Nmin = fenske_Nmin(xD[iLK], xD[iHK], xB[iLK], xB[iHK], alpha_LK_HK)

    # --- Underwood (eigenvalue)
    theta = solve_theta_eigen(alpha_list, zF, specs.q, iLK, iHK)
    Rmin = underwood_Rmin(alpha_list, xD, theta)
    R = specs.R_override if specs.R_override else specs.RR_factor * Rmin

    # --- Gilliland (finite reflux)
    N = gilliland_N(Nmin, R, Rmin)

    # --- Kirkbride (feed stage)
    ratio = kirkbride_ratio(B, D, zF[iLK], zF[iHK], xB[iLK], xD[iHK])
    Ns = N / (1 + ratio)
    Nr = N - Ns
    feed_stage_from_top = int(round(Nr))

    # -----------------------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------------------
    print("\n=== Shortcut Distillation (Eigenvalue Underwood) ===")
    print(f"T_top (bubble)   = {T_top:.2f} K")
    print(f"T_bottom (bubble)= {T_bottom:.2f} K")
    print(f"α(LK/HK)         = {alpha_LK_HK:.5f}")
    print(f"Nmin (Fenske)    = {Nmin:.3f}")
    print(f"θ (Underwood)    = {theta:.6f}")
    print(f"Rmin (Underwood) = {Rmin:.6f}")
    print(f"R (used)         = {R:.4f}")
    print(f"N (Gilliland)    = {N:.2f}")
    print(f"D = {D:.2f} kmol/h,  B = {B:.2f} kmol/h")
    print("xD:", np.round(xD, 5))
    print("xB:", np.round(xB, 5))
    print("\nα_avg (w.r.t. HK):")
    for c in components:
        print(f"  {c:>16}: {alpha_dict[c]:.5f}")
    print(f"\nKirkbride → Nr = {Nr:.2f}, Ns = {Ns:.2f}, Feed stage (from top) ≈ {feed_stage_from_top}")

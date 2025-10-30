
# Shortcut Distillation (B40DI) — NIST Antoine (bar, K)


import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# NIST Antoine data (bar, K)
# log10(P_bar) = A - B / (T + C)
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
# Antoine + K-value functions
# -----------------------------------------------------------------------------
def _pick_antoine_set(component: str, T_K: float):
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
    A, B, C, ref, ok = _pick_antoine_set(component, T_K)
    return 10 ** (A - B / (T_K + C))

def K_value(component: str, T_K: float, P_bar: float) -> float:
    return antoine_psat_bar(component, T_K) / P_bar

# -----------------------------------------------------------------------------
# Bubble point temperature solver
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
# Average relative volatility (ᾱ) w.r.t. HK
# -----------------------------------------------------------------------------
def average_alpha_wrt_HK(components: List[str], HK: str, T_top_K: float, T_bottom_K: float, P_bar: float):
    K_top = np.array([K_value(c, T_top_K, P_bar) for c in components])
    K_bot = np.array([K_value(c, T_bottom_K, P_bar) for c in components])
    iHK = components.index(HK)
    return {c: max(1e-12, np.sqrt(K_top[i] / K_top[iHK] * K_bot[i] / K_bot[iHK])) for i, c in enumerate(components)}

# -----------------------------------------------------------------------------
# Dataclass for design specs
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
# -----------------------------------------------------------------------------
def compute_product_splits(components, zF, s: Specs):
    idx = {c: i for i, c in enumerate(components)}
    iLK, iHK = idx[s.LK], idx[s.HK]
    F = s.F_kmol_h
    d = np.zeros_like(zF)
    b = np.zeros_like(zF)

    F_HK = F * zF[iHK]
    b[iHK] = s.recov_HK_to_B * F_HK
    d[iHK] = F_HK - b[iHK]

    F_LK = F * zF[iLK]
    d[iLK] = 0.98 * F_LK
    b[iLK] = F_LK - d[iLK]

    T_mid = 310.0
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

    D = np.sum(d); B = np.sum(b)
    return d / D, b / B, D, B

# -----------------------------------------------------------------------------
# Shortcut equations
# -----------------------------------------------------------------------------
def fenske_Nmin(xD_LK, xD_HK, xB_LK, xB_HK, alpha):
    return math.log((xD_LK / xD_HK) * (xB_HK / xB_LK)) / math.log(alpha)

# ---------- Underwood ----------
def underwood_phi(theta: float, alphas: List[float], zF: np.ndarray, q: float) -> float:
    """
    Φ(θ) = Σ α_i zF_i / (α_i - θ) - (1 - q)
    Root must lie between α_HK and α_LK.
    """
    a = np.array(alphas); z = np.array(zF, dtype=float)
    return float(np.sum(a * z / (a - theta)) - (1.0 - q))

def solve_theta_bisection(alphas: List[float], zF: np.ndarray, q: float,
                          iLK: int, iHK: int,
                          tol: float = 1e-12, itmax: int = 200) -> float:
    """
    Find θ in (α_HK, α_LK) such that Φ(θ)=0 using robust bisection.
    """
    # initial bracket strictly between α_HK and α_LK
    lo = float(alphas[iHK]) + 1e-12
    hi = float(alphas[iLK]) - 1e-12
    flo = underwood_phi(lo, alphas, zF, q)
    fhi = underwood_phi(hi, alphas, zF, q)

    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        span = 0.05 * max(1.0, abs(hi - lo))
        lo -= span; hi += span
        flo = underwood_phi(lo, alphas, zF, q)
        fhi = underwood_phi(hi, alphas, zF, q)

    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        fmid = underwood_phi(mid, alphas, zF, q)
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)

def underwood_Rmin(alphas: List[float], xD: np.ndarray, theta: float) -> float:
    """
    Underwood Eq. (2) for total condenser:
    Rmin = Σ xD_i α_i/(α_i - θ) - 1
    """
    a = np.array(alphas); xd = np.array(xD, dtype=float)
    return float(np.sum(xd * a / (a - theta)) - 1.0)

def gilliland_N(Nmin, R, Rmin):
    X = (R - Rmin) / (R + 1.0)
    Y = 1.0 - math.exp((1 + 54.4 * X) * (X - 1.0) / 11.0)
    return (Y + Nmin) / (1 - Y)

def kirkbride_ratio(B, D, zF_LK, zF_HK, xB_LK, xD_HK):
    eps = 1e-16
    term = (B / max(D, eps)) * ((zF_HK + eps) / (zF_LK + eps)) * ((xB_LK + eps) / (xD_HK + eps)) ** 2
    return term ** 0.206

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    components = ["i-butane", "n-butane", "1-butene", "trans-2-butene", "cis-2-butene"]
    zF = np.array([0.329, 0.172, 0.264, 0.156, 0.079])
    @dataclass
    class Specs:
        F_kmol_h: float; P_bar: float; q: float
        condenser_type: str; reboiler_type: str
        LK: str; HK: str
        recov_HK_to_B: float; max_LK_in_B_molfrac: float
        RR_factor: float; R_override: Optional[float] = None
    specs = Specs(
        F_kmol_h=1000.0, P_bar=5.0, q=0.2,
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
    iLK = components.index(specs.LK); iHK = components.index(specs.HK)

    # --- Fenske
    alpha_LK_HK = alpha_dict[specs.LK] / alpha_dict[specs.HK]
    Nmin = fenske_Nmin(xD[iLK], xD[iHK], xB[iLK], xB[iHK], alpha_LK_HK)

    # --- Underwood (standard bisection)
    theta = solve_theta_bisection(alpha_list, zF, specs.q, iLK, iHK)
    Rmin = underwood_Rmin(alpha_list, xD, theta)
    R = specs.R_override if specs.R_override else specs.RR_factor * Rmin

    # --- Gilliland
    N = gilliland_N(Nmin, R, Rmin)

    # --- Kirkbride
    ratio = kirkbride_ratio(B, D, zF[iLK], zF[iHK], xB[iLK], xD[iHK])
    Ns = N / (1 + ratio); Nr = N - Ns
    feed_stage_from_top = int(round(Nr))

    # --- OUTPUT
    print("\n=== Shortcut Distillation ===")
    print(f"T_top (bubble)   = {T_top:.2f} K")
    print(f"T_bottom (bubble)= {T_bottom:.2f} K")
    print(f"α(LK/HK)         = {alpha_LK_HK:.5f}")
    print(f"Nmin (Fenske)    = {Nmin:.3f}")
    print(f"θ (Underwood eq1)    = {theta:.6f}")
    print(f"Rmin (Underwood eq2) = {Rmin:.6f}")
    print(f"R (used)         = {R:.4f}")
    print(f"N (Gilliland)    = {N:.2f}")
    print(f"D = {D:.2f} kmol/h,  B = {B:.2f} kmol/h")
    print("xD:", np.round(xD, 5))
    print("xB:", np.round(xB, 5))
    print("\nα_avg (w.r.t. HK):")
    for c in components:
        print(f"  {c:>16}: {alpha_dict[c]:.5f}")
    print(f"\nKirkbride → Nr = {Nr:.2f}, Ns = {Ns:.2f}, Feed stage (from top) ≈ {feed_stage_from_top}")

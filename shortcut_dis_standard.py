# Shortcut Distillation (B40DI) — NIST Antoine (bar, K), simple Underwood + diagnostics
# Flow: Fenske → Underwood (simple) → Gilliland → Kirkbride
# Thermo: Psat from NIST Antoine (bar,K) (multi-range); K = Psat/P; Dew/Bubble in Kelvin
#
# GUI defaults: C4s (i-C4, n-C4, 1-C4=1-butene, t-2-C4, c-2-C4), F=1000 kmol/h, P=10 bar, q=0.2, etc.
# Diagnostics: prints α/K table, θ position, min|α-θ|, and FUGK summary.

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Multi-range NIST Antoine (bar, K)   log10(P_bar) = A - B / (T_K + C)
# Each entry: (Tmin_K, Tmax_K, A, B, C, "reference tag")
# -----------------------------------------------------------------------------
NIST_ANTOINE_RANGES: Dict[str, List[Tuple[float, float, float, float, float, str]]] = {
    # n-Butane — three ranges (K)
    # 135.42–212.89 : 4.70812 1200.475 -13.013
    # 195.11–272.81 : 3.85002 909.65   -36.146
    # 272.66–425.12 : 4.35576 1175.581  -2.071
    "n-butane": [
        (135.42, 212.89, 4.70812, 1200.475, -13.013, "Carruth & Kobayashi (1973)"),
        (195.11, 272.81, 3.85002, 909.65,   -36.146, "Aston & Messerly (1940)"),
        (272.66, 425.12, 4.35576, 1175.581,  -2.071, "Das, Reed, et al. (1973)"),
    ],

    # 1-Butene — one range (K)
    # 195.7–269.4 : 4.24696 1099.207 -8.256
    "1-butene": [
        (195.70, 269.40, 4.24696, 1099.207, -8.256, "Coffin & Maass (1928)"),
    ],

    # Isobutane — two ranges (K)
    # 188.06–261.54 : 3.94417 912.141  -29.808
    # 261.31–408.12 : 4.32810 1132.108   0.918
    "i-butane": [
        (188.06, 261.54, 3.94417, 912.141,  -29.808, "Aston, Kennedy, et al. (1940)"),
        (261.31, 408.12, 4.32810, 1132.108,   0.918, "Das, Reed, et al. (1973)"),
    ],

    # cis-2-Butene — one range (K)
    # 203.06–295.91 : 3.98744 957.06 -36.504
    "cis-2-butene": [
        (203.06, 295.91, 3.98744, 957.06, -36.504, "Scott, Ferguson, et al. (1944)"),
    ],

    # trans-2-Butene — one range (K)
    # 201.70–274.13 : 4.04360 982.166 -30.775
    "trans-2-butene": [
        (201.70, 274.13, 4.04360, 982.166, -30.775, "Guttman & Pitzer (1945)"),
    ],
}


def _pick_antoine_set(component: str, T_K: float) -> Tuple[float, float, float, float, float, str, bool]:
    """
    Select Antoine set for T_K.
    Returns (Tmin, Tmax, A, B, C, ref, in_range). If T_K is outside every range,
    uses the closest midpoint range and returns in_range=False (extrapolation).
    """
    if component not in NIST_ANTOINE_RANGES:
        raise KeyError(f"Component '{component}' not found in NIST_ANTOINE_RANGES.")
    sets = NIST_ANTOINE_RANGES[component]

    for (tmin, tmax, A, B, C, ref) in sets:
        if tmin <= T_K <= tmax:
            return (tmin, tmax, A, B, C, ref, True)

    # Out of range → choose closest midpoint
    mids = [0.5 * (tmin + tmax) for (tmin, tmax, *_rest) in sets]
    idx = int(np.argmin([abs(T_K - m) for m in mids]))
    tmin, tmax, A, B, C, ref = sets[idx]
    return (tmin, tmax, A, B, C, ref, False)


def antoine_psat_bar(component: str, T_K: float) -> float:
    """Saturation pressure (bar) using appropriate NIST range; warns if extrapolating."""
    tmin, tmax, A, B, C, ref, ok = _pick_antoine_set(component, T_K)
    if not ok:
        print(f"⚠️  {component}: T={T_K:.2f} K outside all NIST ranges — extrapolating with "
              f"[{tmin:.2f}, {tmax:.2f}] K ({ref}).")
    return 10.0 ** (A - B / (T_K + C))


def K_value(component: str, T_K: float, P_column_bar: float) -> float:
    """VLE K-value (ideal Raoult): K = Psat / P."""
    Psat = antoine_psat_bar(component, T_K)
    return Psat / max(P_column_bar, 1e-12)


# -----------------------------------------------------------------------------
# Dew / Bubble point solvers (Kelvin)
# Bubble: sum(x_i * K_i(T)) = 1
# Dew:    sum(y_i / K_i(T)) = 1
# -----------------------------------------------------------------------------
def bubble_point_TK(components: List[str], x: np.ndarray, P_bar: float,
                    T_guess_K: float = 300.0) -> float:
    T = float(T_guess_K)
    for _ in range(120):
        K = np.array([K_value(c, T, P_bar) for c in components], dtype=float)
        f = float(np.dot(x, K) - 1.0)
        if abs(f) < 1e-9:
            return T
        dT = 0.5
        Kp = np.array([K_value(c, T + dT, P_bar) for c in components], dtype=float)
        fp = float(np.dot(x, Kp) - 1.0)
        dfdT = (fp - f) / dT
        if abs(dfdT) < 1e-12:
            T += -math.copysign(0.5, f)
        else:
            T -= f / dfdT
        T = max(50.0, T)
    return T


def dew_point_TK(components: List[str], y: np.ndarray, P_bar: float,
                 T_guess_K: float = 300.0) -> float:
    T = float(T_guess_K)
    for _ in range(120):
        K = np.array([K_value(c, T, P_bar) for c in components], dtype=float)
        invK = 1.0 / np.clip(K, 1e-16, None)
        f = float(np.dot(y, invK) - 1.0)
        if abs(f) < 1e-9:
            return T
        dT = 0.5
        Kp = np.array([K_value(c, T + dT, P_bar) for c in components], dtype=float)
        invKp = 1.0 / np.clip(Kp, 1e-16, None)
        fp = float(np.dot(y, invKp) - 1.0)
        dfdT = (fp - f) / dT
        if abs(dfdT) < 1e-12:
            T += -math.copysign(0.5, f)
        else:
            T -= f / dfdT
        T = max(50.0, T)
    return T


# -----------------------------------------------------------------------------
# Shortcut core: Fenske / Underwood (simple) / Gilliland / Kirkbride
# -----------------------------------------------------------------------------
def average_alpha_wrt_HK(components: List[str], HK: str,
                         T_top_K: float, T_bottom_K: float, P_bar: float) -> Dict[str, float]:
    K_top = {c: K_value(c, T_top_K, P_bar) for c in components}
    K_bot = {c: K_value(c, T_bottom_K, P_bar) for c in components}
    alpha_avg = {}
    for c in components:
        a_top = K_top[c] / max(K_top[HK], 1e-16)
        a_bot = K_bot[c] / max(K_bot[HK], 1e-16)
        alpha_avg[c] = max(1e-12, (a_top * a_bot) ** 0.5)
    return alpha_avg


def fenske_Nmin(xD_LK: float, xD_HK: float,
                xB_LK: float, xB_HK: float,
                alpha_LK_HK: float) -> float:
    eps = 1e-16
    term = ((xD_LK + eps) / (xD_HK + eps)) * ((xB_HK + eps) / (xB_LK + eps))
    if alpha_LK_HK <= 1.0:
        alpha_LK_HK = 1.0000001
    return math.log(term, 10) / math.log(alpha_LK_HK, 10)


# -------- Underwood (Part A: simple, robust) --------
@dataclass
class UnderwoodResult:
    theta: float
    Rmin: float
    iterations: int
    bracket: Tuple[float, float]

class UnderwoodError(Exception):
    pass

def _check_prob_vec(vec: List[float], name: str):
    if any(v < 0 for v in vec):
        raise UnderwoodError(f"{name} has negative entries.")
    s = float(np.sum(vec))
    if abs(s - 1.0) > 1e-6:
        raise UnderwoodError(f"{name} must sum to 1.0 (got {s:.6f}).")

def _underwood_f(theta: float, alphas: List[float], zF: List[float], q: float) -> float:
    # f(theta) = sum(alpha_i zF_i / (alpha_i - theta)) - (1 - q)
    total = 0.0
    for a, z in zip(alphas, zF):
        denom = a - theta
        if abs(denom) < 1e-14:
            return float('inf') if denom >= 0 else float('-inf')
        total += (a * z) / denom
    return total - (1.0 - q)

def _bisect_root(func, a: float, b: float, max_iter: int = 200, tol: float = 1e-12):
    fa = func(a); fb = func(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
        raise UnderwoodError("Bisection requires a bracket [a,b] with opposite signs.")
    it = 0
    lo, hi = a, b
    while it < max_iter:
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if abs(fm) < tol or abs(hi - lo) < tol:
            return mid, (it + 1)
        if fa * fm < 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
        it += 1
    return 0.5 * (lo + hi), it

def solve_theta_simple(
    alphas: List[float],
    zF: List[float],
    q: float,
    key_indices: Tuple[int, int],
    bracket: Optional[Tuple[float, float]] = None,
    eps: float = 1e-9,
) -> UnderwoodResult:
    """
    Solve the 1st Underwood equation for theta when 0<q<=1.
    'alphas' are relative volatilities with respect to the heavy key (alpha_HK=1).
    key_indices = (i_LK, i_HK).
    """
    if len(alphas) != len(zF):
        raise UnderwoodError("alphas and zF must be the same length.")
    _check_prob_vec(zF, "zF")

    i_LK, i_HK = key_indices
    a_LK = alphas[i_LK]; a_HK = alphas[i_HK]
    if a_LK <= a_HK:
        raise UnderwoodError("Require alpha_LK > alpha_HK (w.r.t. HK).")

    # Default bracket strictly between HK and LK for 0<q<=1
    lo, hi = bracket if bracket is not None else (a_HK + eps, a_LK - eps)

    # Nudge away from singularities if needed
    uniq = sorted(set(alphas))
    for a in uniq:
        if abs(lo - a) < eps: lo += 10 * eps
        if abs(hi - a) < eps: hi -= 10 * eps

    f = lambda th: _underwood_f(th, alphas, zF, q)
    theta, iters = _bisect_root(f, lo, hi, max_iter=500, tol=1e-13)
    return UnderwoodResult(theta=theta, Rmin=np.nan, iterations=iters, bracket=(lo, hi))

def underwood_Rmin_simple(
    alphas: List[float],
    xD: List[float],
    theta: float
) -> float:
    """
    R_min = Sum_i[ alpha_i * xD_i / (alpha_i - theta) ] - 1
    """
    _check_prob_vec(xD, "xD")
    s = 0.0
    for a, xd in zip(alphas, xD):
        denom = a - theta
        if abs(denom) < 1e-14:
            raise UnderwoodError("theta coincides with a component alpha.")
        s += (a * xd) / denom
    return s - 1.0


def gilliland_N(Nmin: float, R: float, Rmin: float) -> float:
    """Eduljee form of Gilliland correlation."""
    if Nmin < 1e-12:
        return 1.0
    X = (R - Rmin) / max(R + 1.0, 1e-12)
    X = min(1.0, max(0.0, X))
    Y = 1.0 - math.exp((1.0 + 54.4 * X) * (X - 1.0) / 11.0)
    if (1.0 - Y) <= 1e-12:
        return 5.0 * Nmin
    return (Y + Nmin) / (1.0 - Y)


def kirkbride_ratio(B: float, D: float,
                    zF_LK: float, zF_HK: float,
                    xB_LK: float, xD_HK: float) -> float:
    """Nr/Ns = [ (B/D) * (zF_HK/zF_LK) * (xB_LK/xD_HK)^2 ]^0.206"""
    eps = 1e-16
    term = (B / max(D, eps)) * ((zF_HK + eps) / (zF_LK + eps)) * ((xB_LK + eps) / (xD_HK + eps)) ** 2
    return term ** 0.206


# -----------------------------------------------------------------------------
# Product split heuristic (keys specified). Iterate to meet LK-in-B cap.
# -----------------------------------------------------------------------------
@dataclass
class Specs:
    F_kmol_h: float
    P_bar: float
    q: float
    condenser_type: str  # "total" or "partial"
    reboiler_type: str   # "total" or "partial"
    LK: str
    HK: str
    recov_HK_to_B: float        # e.g. 0.95
    max_LK_in_B_molfrac: float  # e.g. 0.02
    RR_factor: float            # e.g. 1.6–1.7
    R_override: Optional[float] = None


def compute_product_splits(components: List[str], zF: np.ndarray, s: Specs) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return xD, xB (mole fractions), and D, B (kmol/h) meeting HK recovery and LK-in-B cap."""
    idx = {c: i for i, c in enumerate(components)}
    iLK, iHK = idx[s.LK], idx[s.HK]
    F = s.F_kmol_h

    d = np.zeros_like(zF)  # to distillate (kmol/h)
    b = np.zeros_like(zF)  # to bottoms   (kmol/h)

    # HK recovery to bottoms
    F_HK = F * zF[iHK]
    b[iHK] = s.recov_HK_to_B * F_HK
    d[iHK] = F_HK - b[iHK]

    # Send most of LK to distillate
    F_LK = F * zF[iLK]
    d[iLK] = 0.98 * F_LK
    b[iLK] = F_LK - d[iLK]

    # Non-keys: light → D, heavy → B, mid → 50/50 (by a mid-column K at ~310K)
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

    # Iterate to satisfy LK-in-bottoms cap
    for _ in range(60):
        D = float(np.sum(d)); B = float(np.sum(b))
        xB = b / max(B, 1e-12)
        if xB[iLK] > s.max_LK_in_B_molfrac:
            delta = 0.2 * (xB[iLK] - s.max_LK_in_B_molfrac) * B
            delta = min(delta, b[iLK])
            b[iLK] -= delta
            d[iLK] += delta
        else:
            break

    D = float(np.sum(d)); B = float(np.sum(b))
    xD = d / max(D, 1e-12)
    xB = b / max(B, 1e-12)
    return xD, xB, D, B


# -----------------------------------------------------------------------------
# End-to-end shortcut workflow
# -----------------------------------------------------------------------------
def shortcut_design(components: List[str], zF: np.ndarray, s: Specs) -> Dict:
    # 1) Product splits
    xD, xB, D, B = compute_product_splits(components, zF, s)

    # 2) End temperatures
    if s.condenser_type == "total":
        T_top_K = bubble_point_TK(components, xD, s.P_bar, T_guess_K=300.0)
    else:
        T_top_K = dew_point_TK(components, xD, s.P_bar, T_guess_K=300.0)
    T_bottom_K = bubble_point_TK(components, xB, s.P_bar, T_guess_K=360.0)

    # 3) Average α wrt HK
    alpha_avg = average_alpha_wrt_HK(components, s.HK, T_top_K, T_bottom_K, s.P_bar)

    # Check LK/HK ordering
    if alpha_avg[s.LK] <= alpha_avg[s.HK]:
        raise ValueError(
            f"Chosen LK/HK inconsistent at this P: α_avg({s.LK})={alpha_avg[s.LK]:.4f} "
            f"≤ α_avg({s.HK})={alpha_avg[s.HK]:.4f}. Swap LK/HK or change P."
        )

    # 4) Fenske Nmin (keys only)
    iLK = components.index(s.LK)
    iHK = components.index(s.HK)
    alpha_LK_HK = alpha_avg[s.LK] / max(alpha_avg[s.HK], 1e-16)
    Nmin = max(0.0, fenske_Nmin(xD[iLK], xD[iHK], xB[iLK], xB[iHK], alpha_LK_HK))

    # 5) Underwood θ and Rmin (simple version)
    alphas_wrt_HK = [alpha_avg[c] for c in components]  # list aligned to components
    zF_list = zF.tolist() if hasattr(zF, "tolist") else list(zF)
    xD_list = xD.tolist() if hasattr(xD, "tolist") else list(xD)

    res_theta = solve_theta_simple(alphas_wrt_HK, zF_list, s.q, (iLK, iHK))
    theta = res_theta.theta
    Rmin_raw = underwood_Rmin_simple(alphas_wrt_HK, xD_list, theta)

    # 6) Choose operating R
    if s.R_override is not None:
        R = max(0.0, float(s.R_override))
    else:
        R = max(0.05, s.RR_factor * max(Rmin_raw, 0.0))  # small floor avoids R≈0 collapse

    # 7) Gilliland → N (theoretical)
    N = gilliland_N(Nmin, R, max(Rmin_raw, 0.0))

    # 8) Kirkbride → feed split / stage
    ratio = kirkbride_ratio(B, D, zF[iLK], zF[iHK], xB[iLK], xD[iHK])
    N_column = max(1.0, N)
    Ns = N_column / (1.0 + ratio)
    Nr = N_column - Ns
    feed_stage_from_top = max(1, int(round(Nr)))

    return {
        "xD": xD.tolist(),
        "xB": xB.tolist(),
        "D_kmol_h": D,
        "B_kmol_h": B,
        "T_top_K": T_top_K,
        "T_bottom_K": T_bottom_K,
        "alpha_avg": {c: float(alpha_avg[c]) for c in components},
        "alpha_LK_HK": float(alpha_LK_HK),
        "Nmin": float(Nmin),
        "Underwood_theta": float(theta),
        "Rmin_raw": float(Rmin_raw),
        "R": float(R),
        "N_theoretical": float(N),
        "Nr": float(Nr),
        "Ns": float(Ns),
        "feed_stage_from_top": int(feed_stage_from_top),
    }


# -----------------------------------------------------------------------------
# Diagnostics helpers (prints α/K table, θ location, FUGK summary)
# -----------------------------------------------------------------------------
def _alpha_table(components, HK, T_top_K, T_bottom_K, P_bar):
    K_top = np.array([K_value(c, T_top_K, P_bar) for c in components], dtype=float)
    K_bot = np.array([K_value(c, T_bottom_K, P_bar) for c in components], dtype=float)
    iHK   = components.index(HK)
    eps   = 1e-16
    a_top = K_top / max(K_top[iHK], eps)
    a_bot = K_bot / max(K_bot[iHK], eps)
    a_avg = np.sqrt(a_top * a_bot)
    return {"K_top": K_top, "K_bot": K_bot, "a_top": a_top, "a_bot": a_bot, "a_avg": a_avg}

def _underwood_diagnostics(components, alpha_avg_dict, theta, q, LK, HK):
    a = np.array([alpha_avg_dict[c] for c in components], dtype=float)
    aLK = alpha_avg_dict[LK]
    aHK = alpha_avg_dict[HK]
    lo_key, hi_key = min(aHK, aLK), max(aHK, aLK)
    mindist = float(np.min(np.abs(a - theta)))
    location = ("between HK & LK" if (lo_key < theta < hi_key)
                else ("below all alphas" if theta < np.min(a)
                else ("above all alphas" if theta > np.max(a)
                else "between non-key alphas")))
    regime = ("q<=0 → θ below all α", "0<q<=1 → θ between HK & LK", "q>1 → θ above all α")
    expected = regime[1] if (0.0 < q <= 1.0) else (regime[0] if q <= 0.0 else regime[2])
    return {
        "a_min": float(np.min(a)), "a_max": float(np.max(a)),
        "aHK": float(aHK), "aLK": float(aLK),
        "theta": float(theta), "theta_location": location,
        "min_abs_alpha_minus_theta": mindist,
        "q_expected_rule": expected,
    }

def format_shortcut_report(components, HK, LK, P_bar, q,
                           xD, xB, D, B,
                           T_top_K, T_bottom_K,
                           alpha_avg_dict,
                           theta, Rmin_raw, R_used,
                           Nmin, N, Nr, Ns, feed_stage_from_top):
    # FIXED: use T_bottom_K (not T_bot_K)
    tab = _alpha_table(components, HK, T_top_K, T_bottom_K, P_bar)
    ud  = _underwood_diagnostics(components, alpha_avg_dict, theta, q, LK, HK)
    lines = []
    lines.append("— Shortcut diagnostics —\n")
    lines.append(f"P = {P_bar:.3f} bar, q = {q:.3f}, LK/HK = {LK}/{HK}\n")
    lines.append(f"Top T (bubble): {T_top_K:.2f} K   |   Bottom T (bubble): {T_bottom_K:.2f} K\n")
    lines.append("\nComponent               K_top      K_bot     α_top(HK)  α_bot(HK)  α_avg(HK)\n")
    for i, c in enumerate(components):
        lines.append(f"{c:22s} {tab['K_top'][i]:9.4f}  {tab['K_bot'][i]:9.4f} "
                     f"{tab['a_top'][i]:10.5f}  {tab['a_bot'][i]:10.5f}  {tab['a_avg'][i]:10.5f}\n")
    lines.append("\nUnderwood:\n")
    lines.append(f"  θ = {ud['theta']:.6f}   |   location: {ud['theta_location']}   |   min|α-θ| = {ud['min_abs_alpha_minus_theta']:.2e}\n")
    lines.append(f"  Rule for q={q:.3f}: {ud['q_expected_rule']}\n")
    lines.append(f"  α(HK)≈{ud['aHK']:.5f}, α(LK)≈{ud['aLK']:.5f}, α-range=[{ud['a_min']:.5f}, {ud['a_max']:.5f}]\n")
    lines.append("\nProducts & Specs:\n")
    lines.append(f"  D = {D:.2f} kmol/h   B = {B:.2f} kmol/h\n")
    lines.append(f"  xD = {np.round(xD,5).tolist()}\n")
    lines.append(f"  xB = {np.round(xB,5).tolist()}\n")
    lines.append("\nFenske / Underwood / Gilliland / Kirkbride:\n")
    lines.append(f"  N_min (keys only)  = {Nmin:.3f}\n")
    lines.append(f"  R_min (raw)        = {Rmin_raw:.6f}\n")
    lines.append(f"  R (used)           = {R_used:.4f}\n")
    lines.append(f"  N (theoretical)    = {N:.2f}\n")
    lines.append(f"  Nr = {Nr:.2f},  Ns = {Ns:.2f},  feed stage (from top) ≈ {feed_stage_from_top}\n")
    return "".join(lines)


# -----------------------------------------------------------------------------
# Tkinter GUI
# -----------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    all_components = sorted(list(NIST_ANTOINE_RANGES.keys()))

    root = tk.Tk()
    root.title("Shortcut Distillation (B40DI) — NIST Antoine (bar,K) — Part A (Simple Underwood)")
    root.resizable(False, False)
    pad = {"padx": 8, "pady": 4}

    # Defaults tailored to C4s (Problem 2 style)
    defaults = {
        "n_comp": 5,
        "components": ["i-butane", "n-butane", "1-butene", "trans-2-butene", "cis-2-butene"],
        "zF": [0.329, 0.172, 0.264, 0.156, 0.079],  # sums to 1.0
        "F": 1000.0,   # kmol/h
        "P": 1.0,     # bar
        "q": 0.2,      # 80% vapor
        "condenser": "total",
        "reboiler": "partial",
        "LK": "1-butene",
        "HK": "n-butane",
        "recov_HK_to_B": 0.95,
        "max_LK_in_B": 0.02,
        "RR_factor": 1.2,
        "R_override": "",
    }

    frm = ttk.Frame(root); frm.grid(row=0, column=0, sticky="nsew", **pad)

    ttk.Label(frm, text="Number of components").grid(row=0, column=0, sticky="w", **pad)
    n_var = tk.StringVar(value=str(defaults["n_comp"]))
    ttk.Entry(frm, textvariable=n_var, width=8).grid(row=0, column=1, **pad)

    ttk.Label(frm, text="Components (comma-separated)").grid(row=1, column=0, sticky="w", **pad)
    comps_var = tk.StringVar(value=", ".join(defaults["components"]))
    ttk.Entry(frm, textvariable=comps_var, width=48).grid(row=1, column=1, columnspan=3, **pad)

    ttk.Label(frm, text="Feed mole fractions zF (comma-separated)").grid(row=2, column=0, sticky="w", **pad)
    zf_var = tk.StringVar(value=", ".join(map(str, defaults["zF"])))
    ttk.Entry(frm, textvariable=zf_var, width=48).grid(row=2, column=1, columnspan=3, **pad)

    ttk.Label(frm, text="Feed flow F (kmol/h)").grid(row=3, column=0, sticky="w", **pad)
    F_var = tk.StringVar(value=str(defaults["F"]))
    ttk.Entry(frm, textvariable=F_var, width=12).grid(row=3, column=1, **pad)

    ttk.Label(frm, text="Column pressure P (bar)").grid(row=3, column=2, sticky="w", **pad)
    P_var = tk.StringVar(value=str(defaults["P"]))
    ttk.Entry(frm, textvariable=P_var, width=12).grid(row=3, column=3, **pad)

    ttk.Label(frm, text="Feed quality q (0=dew vap, 1=bubble liq)").grid(row=4, column=0, sticky="w", **pad)
    q_var = tk.StringVar(value=str(defaults["q"]))
    ttk.Entry(frm, textvariable=q_var, width=12).grid(row=4, column=1, **pad)

    ttk.Label(frm, text="Condenser type").grid(row=4, column=2, sticky="w", **pad)
    cond_var = tk.StringVar(value=defaults["condenser"])
    ttk.Combobox(frm, textvariable=cond_var, values=["total", "partial"], state="readonly", width=10).grid(row=4, column=3, **pad)

    ttk.Label(frm, text="Reboiler type").grid(row=5, column=2, sticky="w", **pad)
    reb_var = tk.StringVar(value=defaults["reboiler"])
    ttk.Combobox(frm, textvariable=reb_var, values=["total", "partial"], state="readonly", width=10).grid(row=5, column=3, **pad)

    ttk.Label(frm, text="Light key (LK)").grid(row=5, column=0, sticky="w", **pad)
    LK_var = tk.StringVar(value=defaults["LK"])
    ttk.Combobox(frm, textvariable=LK_var, values=all_components, width=20).grid(row=5, column=1, **pad)

    ttk.Label(frm, text="Heavy key (HK)").grid(row=6, column=0, sticky="w", **pad)
    HK_var = tk.StringVar(value=defaults["HK"])
    ttk.Combobox(frm, textvariable=HK_var, values=all_components, width=20).grid(row=6, column=1, **pad)

    ttk.Label(frm, text="HK recovery to bottoms (0–1)").grid(row=6, column=2, sticky="w", **pad)
    recHK_var = tk.StringVar(value=str(defaults["recov_HK_to_B"]))
    ttk.Entry(frm, textvariable=recHK_var, width=12).grid(row=6, column=3, **pad)

    ttk.Label(frm, text="Max LK mole fraction in bottoms").grid(row=7, column=2, sticky="w", **pad)
    maxLK_var = tk.StringVar(value=str(defaults["max_LK_in_B"]))
    ttk.Entry(frm, textvariable=maxLK_var, width=12).grid(row=7, column=3, **pad)

    ttk.Label(frm, text="R/Rmin factor (e.g. 1.5–1.7)").grid(row=7, column=0, sticky="w", **pad)
    RR_var = tk.StringVar(value=str(defaults["RR_factor"]))
    ttk.Entry(frm, textvariable=RR_var, width=12).grid(row=7, column=1, **pad)

    ttk.Label(frm, text="Override R (blank = use factor)").grid(row=8, column=0, sticky="w", **pad)
    Rabs_var = tk.StringVar(value=defaults["R_override"])
    ttk.Entry(frm, textvariable=Rabs_var, width=12).grid(row=8, column=1, **pad)

    out = tk.Text(root, height=26, width=112)
    out.grid(row=1, column=0, sticky="nsew", **pad)

    def run_calc():
        try:
            n = int(n_var.get())
            comps = [c.strip() for c in comps_var.get().split(",") if c.strip()]
            if len(comps) != n:
                raise ValueError("Number of components does not match the list length.")
            for c in comps:
                if c not in NIST_ANTOINE_RANGES:
                    raise ValueError(f"Unknown component: '{c}'. Available: {', '.join(sorted(NIST_ANTOINE_RANGES.keys()))}")

            zF_list = [float(x.strip()) for x in zf_var.get().split(",") if len(x.strip()) > 0]
            if len(zF_list) != n:
                raise ValueError("zF length must equal number of components.")
            zF_arr = np.array(zF_list, dtype=float)
            if min(zF_arr) < 0 or abs(np.sum(zF_arr) - 1.0) > 1e-6:
                raise ValueError("Feed mole fractions must be non-negative and sum to 1.0")

            F = float(F_var.get())
            P = float(P_var.get())
            q = float(q_var.get())
            condenser = cond_var.get()
            reboiler = reb_var.get()
            LK = LK_var.get()
            HK = HK_var.get()
            recHK = float(recHK_var.get())
            maxLK = float(maxLK_var.get())
            RR = float(RR_var.get())
            R_override = None if len(Rabs_var.get().strip()) == 0 else float(Rabs_var.get().strip())

            if LK not in comps or HK not in comps:
                raise ValueError("LK and HK must be among the chosen components.")
            if not (0.0 < recHK < 1.0):
                raise ValueError("HK recovery to bottoms should be between 0 and 1.")
            if maxLK < 0 or maxLK > 0.2:
                raise ValueError("Max LK in bottoms should be sensible (e.g. 0–0.05).")
            if RR <= 0 and R_override is None:
                raise ValueError("Provide a positive R/Rmin factor or override R.")

            s = Specs(
                F_kmol_h=F, P_bar=P, q=q,
                condenser_type=condenser, reboiler_type=reboiler,
                LK=LK, HK=HK, recov_HK_to_B=recHK, max_LK_in_B_molfrac=maxLK,
                RR_factor=RR, R_override=R_override
            )
            res = shortcut_design(comps, zF_arr, s)

            # --- Tidy summary (existing block) ---
            out.delete("1.0", "end")
            out.insert("end", "=== Shortcut Design Results (NIST bar/K) — Part A (Simple Underwood) ===\n")
            out.insert("end", f"Components: {comps}\n")
            out.insert("end", f"Feed zF:    {np.round(zF_arr, 4).tolist()}\n")
            out.insert("end", f"Pressure:   {P:.3f} bar,  q = {q:.3f}\n")
            out.insert("end", f"Condenser:  {condenser}, Reboiler: {reboiler}\n")
            out.insert("end", f"LK/HK:      {LK} / {HK}\n")
            out.insert("end", f"T_top:      {res['T_top_K']:.2f} K\n")
            out.insert("end", f"T_bottom:   {res['T_bottom_K']:.2f} K\n\n")

            out.insert("end", f"D (kmol/h): {res['D_kmol_h']:.2f}\n")
            out.insert("end", f"B (kmol/h): {res['B_kmol_h']:.2f}\n")
            out.insert("end", f"xD:         {np.round(np.array(res['xD']), 5).tolist()}\n")
            out.insert("end", f"xB:         {np.round(np.array(res['xB']), 5).tolist()}\n\n")

            out.insert("end", f"α_avg (wrt {HK}):\n")
            for c in comps:
                out.insert("end", f"  {c:>16s}: {res['alpha_avg'][c]:.5f}\n")
            out.insert("end", f"\nα(LK/HK):   {res['alpha_LK_HK']:.5f}\n")
            out.insert("end", f"Nmin:       {res['Nmin']:.3f}\n")

            out.insert("end", f"Underwood θ: {res['Underwood_theta']:.6f}\n")
            out.insert("end", f"Rmin (raw):  {res['Rmin_raw']:.6f}\n")
            out.insert("end", f"R (used):    {res['R']:.4f}\n")
            out.insert("end", f"N (theor.):  {res['N_theoretical']:.2f}\n\n")

            out.insert("end", f"Kirkbride → Nr: {res['Nr']:.2f},  Ns: {res['Ns']:.2f}\n")
            out.insert("end", f"Estimated feed stage (from top): {res['feed_stage_from_top']}\n")

            # --- Diagnostics table ---
            diag_text = format_shortcut_report(
                components=comps, HK=HK, LK=LK, P_bar=P, q=q,
                xD=np.array(res['xD']), xB=np.array(res['xB']),
                D=res['D_kmol_h'], B=res['B_kmol_h'],
                T_top_K=res['T_top_K'], T_bottom_K=res['T_bottom_K'],
                alpha_avg_dict=res['alpha_avg'],
                theta=res['Underwood_theta'],
                Rmin_raw=res['Rmin_raw'], R_used=res['R'],
                Nmin=res['Nmin'], N=res['N_theoretical'],
                Nr=res['Nr'], Ns=res['Ns'],
                feed_stage_from_top=res['feed_stage_from_top'],
            )
            out.insert("end", "\n" + diag_text + "\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    btns = ttk.Frame(frm); btns.grid(row=9, column=0, columnspan=4, sticky="e", **pad)
    ttk.Button(btns, text="Run Shortcut Design", command=run_calc).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="Quit", command=root.destroy).grid(row=0, column=1, padx=6)

    root.mainloop()

if __name__ == "__main__":
    run_gui()

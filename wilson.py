import json
import math
import argparse
from typing import Optional, Dict

# =========================
# Built-in data
# =========================
#
# Antoine constants (bar/K; base-10): log10(P[bar]) = A - B/(T[K] + C)
# Using your supplied A/B/C exactly, in bar/K, base-10.
ANTOINE_BAR_K = {
    # --- Gases / light species ---
    "ethylene": {"A": 3.87261, "B": 584.146,  "C": -18.307, "Tmin": 0.0, "Tmax": 1000.0},
    "chlorine": {"A": 4.28814, "B": 969.992,  "C": -12.791, "Tmin": 0.0, "Tmax": 1000.0},
    "oxygen":   {"A": 3.85845, "B": 325.675,  "C": -5.667,  "Tmin": 0.0, "Tmax": 1000.0},
    "nitrogen": {"A": 3.7362,  "B": 264.651,  "C": -6.788,  "Tmin": 0.0, "Tmax": 1000.0},
    # Note: "Hydrochloric Acid" data are commonly used for anhydrous HCl vapor;
    # keep your Antoine constants as provided.
    "hydrochloric acid": {"A": 4.57389, "B": 868.358,  "C": 1.754,   "Tmin": 0.0, "Tmax": 1000.0},
    "hydrogen chloride": {"A": 4.57389, "B": 868.358,  "C": 1.754,   "Tmin": 0.0, "Tmax": 1000.0},  # alias

    # --- Chlorinated organics & water/CO2 ---
    "1,2-dichloroethane":      {"A": 4.58518, "B": 1521.789, "C": -24.67,  "Tmin": 0.0, "Tmax": 1000.0},
    "1,1,2-trichloroethane":   {"A": 4.06974, "B": 1310.297, "C": -64.41,  "Tmin": 0.0, "Tmax": 1000.0},
    "water":                   {"A": 8.0713,  "B": 1730.63,  "C": 233.43,  "Tmin": 0.0, "Tmax": 1000.0},
    "carbon dioxide":          {"A": 6.9758,  "B": 1347.79,  "C": 273.0,   "Tmin": 0.0, "Tmax": 1000.0},
    "vinyl chloride monomer":  {"A": 3.98598, "B": 892.757,  "C": -35.051, "Tmin": 0.0, "Tmax": 1000.0},
    "vinyl chloride":          {"A": 3.98598, "B": 892.757,  "C": -35.051, "Tmin": 0.0, "Tmax": 1000.0},  # alias
    "chloroethene":            {"A": 3.98598, "B": 892.757,  "C": -35.051, "Tmin": 0.0, "Tmax": 1000.0},  # alias
    "acetylene":               {"A": 4.66141, "B": 909.079,  "C": 7.947,   "Tmin": 0.0, "Tmax": 1000.0},
    "butyne":                  {"A": 4.2402,  "B": 1046.879, "C": -33.951, "Tmin": 0.0, "Tmax": 1000.0},   # assumed 1-butyne
    "1,2-dichloroethylene":    {"A": 4.069,   "B": 1163.729, "C": -47.174, "Tmin": 0.0, "Tmax": 1000.0},  # assumed trans-
}

# Critical properties & acentric factors for Wilson (Tc [K], Pc [bar], omega).
# Tc/Pc prioritized from NIST WebBook & government/standards docs in bar/K.
# ω from standard tables (Pitzer/Lee–Kesler) or CoolProp where needed.
CRIT_DB = {
    # Light gases
    "ethylene": {"Tc_K": 282.34, "Pc_bar": 50.41, "acentric": 0.089},      # NIST Tc/Pc; ω tables
    "chlorine": {"Tc_K": 416.96, "Pc_bar": 79.91, "acentric": 0.070},      # NIST Tc/Pc; ω typical table value
    "oxygen":   {"Tc_K": 154.58, "Pc_bar": 50.43, "acentric": 0.022},      # NIST Tc/Pc; ω tables
    "nitrogen": {"Tc_K": 126.19, "Pc_bar": 33.978,"acentric": 0.040},      # NIST Tc/Pc; ω tables
    "hydrogen chloride": {"Tc_K": 324.6, "Pc_bar": 83.1,  "acentric": 0.12965}, # standard Tc/Pc; ω CoolProp
    "hydrochloric acid": {"Tc_K": 324.6, "Pc_bar": 83.1,  "acentric": 0.12965}, # alias to HCl

    # Chlorinated organics & others
    "1,2-dichloroethane":    {"Tc_K": 561.6, "Pc_bar": 53.8, "acentric": 0.210},  # Tc/Pc: NIST/Wiki; ω typical handbook
    "1,1,2-trichloroethane": {"Tc_K": 631.0, "Pc_bar": 45.7, "acentric": 0.300},  # standard table values (commonly cited)
    "water":                 {"Tc_K": 647.096, "Pc_bar": 220.64, "acentric": 0.344}, # NIST WTT; ω standard
    "carbon dioxide":        {"Tc_K": 304.1282,"Pc_bar": 73.773, "acentric": 0.228}, # NIST; ω standard
    "vinyl chloride monomer":{"Tc_K": 431.6, "Pc_bar": 53.4, "acentric": 0.122},   # NOAA CHRIS; ω tables
    "vinyl chloride":        {"Tc_K": 431.6, "Pc_bar": 53.4, "acentric": 0.122},   # alias
    "chloroethene":          {"Tc_K": 431.6, "Pc_bar": 53.4, "acentric": 0.122},   # alias
    "acetylene":             {"Tc_K": 308.3, "Pc_bar": 61.4, "acentric": 0.190},   # NIST; ω standard
    # Assumptions explained in header notes:
    "butyne":                {"Tc_K": 426.0, "Pc_bar": 41.0, "acentric": 0.250},   # assumed 1-butyne (typical literature values)
    "1,2-dichloroethylene":  {"Tc_K": 516.0, "Pc_bar": 50.0, "acentric": 0.220},   # assumed trans- isomer, representative values
}

# =========================
# Utility functions
# =========================
def antoine_psat_bar(component: str, T_K: float) -> float:
    comp = component.lower()
    if comp not in ANTOINE_BAR_K:
        raise KeyError(f"Antoine constants for '{component}' not found. Add to ANTOINE_BAR_K.")
    A = ANTOINE_BAR_K[comp]["A"]
    B = ANTOINE_BAR_K[comp]["B"]
    C = ANTOINE_BAR_K[comp]["C"]
    return 10 ** (A - B / (T_K + C))

def k_raoult(component: str, T_K: float, P_bar: float) -> float:
    return antoine_psat_bar(component, T_K) / P_bar

def k_wilson(component: str, T_K: float, P_bar: float) -> float:
    comp = component.lower()
    if comp not in CRIT_DB:
        raise KeyError(f"Critical data for '{component}' not in database. Add to CRIT_DB.")
    Tc = CRIT_DB[comp]["Tc_K"]
    Pc = CRIT_DB[comp]["Pc_bar"]
    omega = CRIT_DB[comp]["acentric"]
    # Wilson (1968): ln K = ln(Pc/P) + 5.373(1+ω)(1 − Tc/T)
    return math.exp(math.log(Pc / P_bar) + 5.373 * (1.0 + omega) * (1.0 - Tc / T_K))

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def k_value(component: str, T: float, P: float, method: str = "wilson",
            units: str = "barK", coeffs_path: Optional[str] = None) -> float:
    if units != "barK":
        raise ValueError("This script expects P in bar and T in K (units='barK').")
    if method == "raoult":
        return k_raoult(component, T, P)
    elif method == "wilson":
        return k_wilson(component, T, P)
    else:
        raise ValueError("Unknown method. Use 'raoult' or 'wilson'.")

def main():
    ap = argparse.ArgumentParser(description="Compute K-values (Raoult/Wilson).")
    ap.add_argument("--component", required=True,
                    help="e.g., ethylene, chlorine, vinyl chloride, 1,2-dichloroethane, etc.")
    ap.add_argument("--T", type=float, required=True, help="Temperature [K]")
    ap.add_argument("--P", type=float, required=True, help="Pressure [bar]")
    ap.add_argument("--method", choices=["raoult", "wilson"], default="wilson")
    args = ap.parse_args()
    K = k_value(args.component, args.T, args.P, method=args.method)
    print(f"K({args.component}, T={args.T} K, P={args.P} bar, method={args.method}) = {K:.6g}")

if __name__ == "__main__":
    main()

import json
import math
import argparse
from typing import Optional, Dict

# =========================
# Built-in data
# =========================
# Antoine constants (bar/K; base-10): log10(P[bar]) = A - B/(T[K] + C)
# Using your supplied A/B/C exactly, in bar/K, base-10.
ANTOINE_BAR_K = {
    # --- Gases / light species ---
    # Range (NIST, bar/K row that matches these A/B/C): 149.37–188.57 K
    "ethylene": {"A": 3.87261, "B": 584.146,  "C": -18.307, "Tmin": 149.37, "Tmax": 188.57},  # NIST Michels & Wassenaar

    # Chlorine has multiple Antoine rows; these A/B/C correspond to the high-T set.
    # Range (NIST): 239.4–400.3 K
    "chlorine": {"A": 4.28814, "B": 969.992,  "C": -12.791, "Tmin": 239.4, "Tmax": 400.3},    # NIST Stull high-T set

    # Range (NIST): 54.36–100.16 K (row that matches these A/B/C)
    "oxygen":   {"A": 3.85845, "B": 325.675,  "C": -5.667,  "Tmin": 54.36, "Tmax": 100.16},   # NIST Brower & Thodos

    # Range (NIST): 63.14–126.0 K (row that matches these A/B/C)
    "nitrogen": {"A": 3.7362,  "B": 264.651,  "C": -6.788,  "Tmin": 63.14, "Tmax": 126.0},    # NIST Edejer & Thodos

    # Note: “Hydrochloric Acid” here refers to anhydrous HCl vapour properties (alias to hydrogen chloride).
    # Range (NIST, high-T row that matches these A/B/C): 188.3–309.4 K
    "hydrochloric acid": {"A": 4.57389, "B": 868.358,  "C": 1.754,   "Tmin": 188.3, "Tmax": 309.4},
    "hydrogen chloride": {"A": 4.57389, "B": 868.358,  "C": 1.754,   "Tmin": 188.3, "Tmax": 309.4},  # alias

    # --- Chlorinated organics & others ---
    # Range (NIST): 242.33–372.60 K
    "1,2-dichloroethane":      {"A": 4.58518, "B": 1521.789, "C": -24.67,  "Tmin": 242.33, "Tmax": 372.60},

    # Range (NIST): 323.12–386.82 K
    "1,1,2-trichloroethane":   {"A": 4.06974, "B": 1310.297, "C": -64.41,  "Tmin": 323.12, "Tmax": 386.82},

    # Water note: your A/B/C correspond to the classic 1–100 °C set (°C, mmHg in many sources).
    # Here we keep your A/B/C as-is (per your request); the **range only** is expressed in kelvin: 274.15–373.15 K.
    "water":                   {"A": 8.0713,  "B": 1730.63,  "C": 233.43,  "Tmin": 274.15, "Tmax": 373.15},

    # CO2 note: your A/B/C are not the same as the NIST bar/K set.
    # We include the NIST bar/K range (154.26–195.89 K) as a conservative validity check.
    "carbon dioxide":          {"A": 6.9758,  "B": 1347.79,  "C": 273.0,   "Tmin": 154.26, "Tmax": 195.89},

    # Range (NIST): 163.13–252.75 K
    "vinyl chloride monomer":  {"A": 3.98598, "B": 892.757,  "C": -35.051, "Tmin": 163.13, "Tmax": 252.75},
    "vinyl chloride":          {"A": 3.98598, "B": 892.757,  "C": -35.051, "Tmin": 163.13, "Tmax": 252.75},  # alias
    "chloroethene":            {"A": 3.98598, "B": 892.757,  "C": -35.051, "Tmin": 163.13, "Tmax": 252.75},  # alias

    # Range (NIST, matching row): 214.64–308.33 K
    "acetylene":               {"A": 4.66141, "B": 909.079,  "C": 7.947,   "Tmin": 214.64, "Tmax": 308.33},

    # “butyne” assumed 1-butyne; range (NIST): 150.20–201.63 K
    "butyne":                  {"A": 4.2402,  "B": 1046.879, "C": -33.951, "Tmin": 150.20, "Tmax": 201.63},

    # Your coefficients match the (Z) (cis) isomer; range (NIST): 273.91–356.78 K
    "1,2-dichloroethylene":    {"A": 4.069,   "B": 1163.729, "C": -47.174, "Tmin": 273.91, "Tmax": 356.78},
}

# Critical properties & acentric factors for Wilson (Tc [K], Pc [bar], omega).
# Tc/Pc prioritised from NIST-style data; ω from standard Pitzer/Lee–Kesler tables or CoolProp where needed.
CRIT_DB = {
    # Light gases
    "ethylene": {"Tc_K": 282.34, "Pc_bar": 50.41,  "acentric": 0.089},
    "chlorine": {"Tc_K": 416.96, "Pc_bar": 79.91,  "acentric": 0.070},
    "oxygen":   {"Tc_K": 154.58, "Pc_bar": 50.43,  "acentric": 0.022},
    "nitrogen": {"Tc_K": 126.19, "Pc_bar": 33.978, "acentric": 0.040},
    "hydrogen chloride": {"Tc_K": 324.6, "Pc_bar": 83.1, "acentric": 0.12965},
    "hydrochloric acid": {"Tc_K": 324.6, "Pc_bar": 83.1, "acentric": 0.12965},  # alias to HCl

    # Chlorinated organics & others
    "1,2-dichloroethane":    {"Tc_K": 561.6, "Pc_bar": 53.8,  "acentric": 0.210},
    "1,1,2-trichloroethane": {"Tc_K": 631.0, "Pc_bar": 45.7,  "acentric": 0.300},
    "water":                 {"Tc_K": 647.096, "Pc_bar": 220.64, "acentric": 0.344},
    "carbon dioxide":        {"Tc_K": 304.1282,"Pc_bar": 73.773, "acentric": 0.228},
    "vinyl chloride monomer":{"Tc_K": 431.6, "Pc_bar": 53.4,  "acentric": 0.122},
    "vinyl chloride":        {"Tc_K": 431.6, "Pc_bar": 53.4,  "acentric": 0.122},  # alias
    "chloroethene":          {"Tc_K": 431.6, "Pc_bar": 53.4,  "acentric": 0.122},  # alias
    "acetylene":             {"Tc_K": 308.3, "Pc_bar": 61.4,  "acentric": 0.190},
    # Assumptions explained previously:
    "butyne":                {"Tc_K": 426.0, "Pc_bar": 41.0,  "acentric": 0.250},  # assumed 1-butyne
    "1,2-dichloroethylene":  {"Tc_K": 516.0, "Pc_bar": 50.0,  "acentric": 0.220},  # assumed trans-
}

# =========================
# Utility functions
# =========================

#Returns PSat in bar
def antoine_psat_bar(component: str, T_K: float) -> float:
    comp = component.lower()
    if comp not in ANTOINE_BAR_K:
        raise KeyError(f"Antoine constants for '{component}' not found. Add to ANTOINE_BAR_K.")
    A = ANTOINE_BAR_K[comp]["A"]
    B = ANTOINE_BAR_K[comp]["B"]
    C = ANTOINE_BAR_K[comp]["C"]
    return 10 ** (A - B / (T_K + C))

# Raoult's method
def k_raoult(component: str, T_K: float, P_bar: float) -> float:
    return antoine_psat_bar(component, T_K) / P_bar

# Wilson's method
def k_wilson(component: str, T_K: float, P_bar: float) -> float:
    comp = component.lower()
    if comp not in CRIT_DB:
        # show close names if there’s a typo
        known = ", ".join(sorted(CRIT_DB.keys()))
        raise KeyError(f"Critical data for '{component}' not in database. Try one of: {known}")
    Tc = CRIT_DB[comp]["Tc_K"]
    Pc = CRIT_DB[comp]["Pc_bar"]
    omega = CRIT_DB[comp]["acentric"]
    # Wilson (1968): ln K = ln(Pc/P) + 5.373(1+ω)(1 − Tc/T)
    return math.exp(math.log(Pc / P_bar) + 5.373 * (1.0 + omega) * (1.0 - Tc / T_K))

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def k_value(component: str, T: float, P: float, method: str = "wilson",
            units: str = "barK") -> float:
    if units != "barK":
        raise ValueError("This script expects P in bar and T in K (units='barK').")
    if method == "raoult":
        return k_raoult(component, T, P)
    elif method == "wilson":
        return k_wilson(component, T, P)
    else:
        raise ValueError("Unknown method. Use 'raoult' or 'wilson'.")

def main():
    ap = argparse.ArgumentParser(description="Compute K-values (Wilson/Raoult).")
    ap.add_argument("--component", required=True,
                    help="e.g., ethylene, chlorine, vinyl chloride, 1,2-dichloroethane, acetylene, etc.")
    ap.add_argument("--T", type=float, required=True, help="Temperature [K]")
    ap.add_argument("--P", type=float, required=True, help="Pressure [bar]")
    ap.add_argument("--method", choices=["raoult", "wilson"], default="wilson")
    args = ap.parse_args()
    K = k_value(args.component, args.T, args.P, method=args.method)
    print(f"K({args.component}, T={args.T} K, P={args.P} bar, method={args.method}) = {K:.6g}")

    # Added: warn if the calculation temperature is outside the Antoine validity range for this component.
    comp = args.component.lower()
    if comp in ANTOINE_BAR_K:
        Tmin = ANTOINE_BAR_K[comp].get("Tmin", None)
        Tmax = ANTOINE_BAR_K[comp].get("Tmax", None)
        if (Tmin is not None and args.T < Tmin) or (Tmax is not None and args.T > Tmax):
            print(f"WARNING: T={args.T} K is outside the Antoine validity range for {args.component} "
                  f"[{Tmin}–{Tmax}] K. Raoult-based Psat/K-values may be unreliable.")

if __name__ == "__main__":
    main()

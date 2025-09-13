import json
import math
import argparse
from typing import Optional, Dict

# =========================
# Built-in data
# =========================

# Antoine constants (bar/K; base-10): log10(P[bar]) = A - B/(T[K] + C)
# Ranges selected to cover ~1 bar region; check NIST for other ranges if needed.
ANTOINE_BAR_K = {
    "n-pentane": {"A": 3.9892, "B": 1070.617, "C": -40.454, "Tmin": 268.8, "Tmax": 341.37},
    "n-hexane":  {"A": 4.00266, "B": 1171.53,  "C": -48.784, "Tmin": 287.9,  "Tmax": 369.0},
    "n-heptane": {"A": 4.02832, "B": 1268.636, "C": -56.199, "Tmin": 299.07, "Tmax": 372.43},
    "n-octane":  {"A": 4.04867, "B": 1355.126, "C": -62.540, "Tmin": 312.0,  "Tmax": 398.0},
    "benzene":   {"A": 4.03238, "B": 1203.531, "C": -53.226, "Tmin": 278.7,  "Tmax": 353.2},
    "toluene":   {"A": 4.07827, "B": 1343.943, "C": -53.773, "Tmin": 294.0,  "Tmax": 404.0},
    # Add more as needed
}

# Critical properties & acentric factors (Tc [K], Pc [bar], omega) for Wilson/DePriester-style use
CRIT_DB = {
    "methane":   {"Tc_K": 190.56, "Pc_bar": 45.99, "acentric": 0.011},
    "ethane":    {"Tc_K": 305.32, "Pc_bar": 48.72, "acentric": 0.099},
    "propane":   {"Tc_K": 369.83, "Pc_bar": 42.48, "acentric": 0.152},
    "n-butane":  {"Tc_K": 425.12, "Pc_bar": 37.96, "acentric": 0.193},
    "isobutane": {"Tc_K": 408.14, "Pc_bar": 36.49, "acentric": 0.176},
    "n-pentane": {"Tc_K": 469.70, "Pc_bar": 33.70, "acentric": 0.251},
    "n-hexane":  {"Tc_K": 507.60, "Pc_bar": 30.25, "acentric": 0.301},
    "n-heptane": {"Tc_K": 540.20, "Pc_bar": 27.36, "acentric": 0.350},
    "n-octane":  {"Tc_K": 568.70, "Pc_bar": 24.97, "acentric": 0.398},
    "n-nonane":  {"Tc_K": 594.60, "Pc_bar": 22.94, "acentric": 0.444},
    "benzene":   {"Tc_K": 562.02, "Pc_bar": 48.94, "acentric": 0.212},
    "toluene":   {"Tc_K": 591.75, "Pc_bar": 41.06, "acentric": 0.262},
    "ethylene":  {"Tc_K": 282.34, "Pc_bar": 50.41, "acentric": 0.089},
    "propene":   {"Tc_K": 364.21, "Pc_bar": 45.60, "acentric": 0.143},
    "isopentane":{"Tc_K": 460.40, "Pc_bar": 33.83, "acentric": 0.227},
    "neopentane":{"Tc_K": 433.80, "Pc_bar": 31.00, "acentric": 0.196},
}

# =========================
# Utility functions
# =========================

def antoine_psat_bar(component: str, T_K: float) -> float:
    comp = component.lower()
    if comp not in ANTOINE_BAR_K:
        raise KeyError(f"Antoine constants for '{component}' not found. Add to ANTOINE_BAR_K or use another method.")
    A, B, C = ANTOINE_BAR_K[comp]["A"], ANTOINE_BAR_K[comp]["B"], ANTOINE_BAR_K[comp]["C"]
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
    # Wilson correlation (1968): ln K = ln(Pc/P) + 5.373(1+ω)(1 − Tc/T)
    return math.exp(math.log(Pc / P_bar) + 5.373*(1.0 + omega)*(1.0 - Tc / T_K))

# ---- McWilliams (1973) DePriester regression ----
# ln(K) = a_T1*(1/Tr) + a_T2*(1/Tr^2) + a_Tln*ln(Tr) + a_T*Tr + a_T2p*(Tr^2) + a_T3*(Tr^3)
#         + a_Pln*ln(Pr) + a_P*Pr + a_P2*(Pr^2)
# with an acentric correction:
#   K(ω) = K(ω=0.27) * 10**( D(Tr) * (ω - 0.27) )
# where D(Tr) = Σ d_i * Tr^i (i=0..6)
#
# You must provide a JSON with the 'a' coefficients for the applicable range/case (Table II)
# and component criticals (Tc, Pc, ω). See mcwilliams_template.json generated alongside.

def k_depriester_mcwilliams(component: str, T_K: float, P_bar: float, coeffs: Dict) -> float:
    comp = component.lower()
    if comp not in coeffs["components"]:
        raise KeyError(f"'{component}' not in McWilliams coefficients file. Add it under coeffs['components'].")
    cdata = coeffs["components"][comp]
    Tc = cdata["Tc_K"]
    Pc_bar = cdata["Pc_bar"]
    omega = cdata["acentric"]
    Tr = T_K / Tc
    Pr = P_bar / Pc_bar

    a = coeffs["regression"]["a"]
    lnK0 = (
        a["T1"] * (1.0/Tr) +
        a["T2"] * (1.0/(Tr**2)) +
        a["Tln"] * math.log(Tr) +
        a["T"] * Tr +
        a["T2p"] * (Tr**2) +
        a["T3"] * (Tr**3) +
        a["Pln"] * math.log(Pr) +
        a["P"] * Pr +
        a["P2"] * (Pr**2)
    )

    Dcoef = coeffs["regression"]["Dpoly"]
    D = 0.0
    pwr = 1.0
    # compute polynomial D(Tr) = d0 + d1 Tr + ... + d6 Tr^6
    for i in range(0, 7):
        di = Dcoef.get(str(i), 0.0)
        if i == 0:
            pwr = 1.0
        elif i == 1:
            pwr = Tr
        else:
            pwr = pwr * Tr
        D += di * pwr

    K = math.exp(lnK0) * (10 ** (D * (omega - 0.27)))
    return K

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def k_value(component: str, T: float, P: float, method: str = "raoult",
            units: str = "barK", coeffs_path: Optional[str] = None) -> float:
    if units != "barK":
        raise ValueError("This script expects P in bar and T in K (units='barK').")

    if method == "raoult":
        return k_raoult(component, T, P)
    elif method == "wilson":
        return k_wilson(component, T, P)
    elif method == "depriester_mcwilliams":
        if coeffs_path is None:
            raise ValueError("Provide coeffs_path (JSON) for depriester_mcwilliams.")
        coeffs = load_json(coeffs_path)
        return k_depriester_mcwilliams(component, T, P, coeffs)
    else:
        raise ValueError("Unknown method. Use 'raoult', 'wilson', or 'depriester_mcwilliams'.")

def main():
    ap = argparse.ArgumentParser(description="Compute K-values (DePriester-style or Raoult/Wilson).")
    ap.add_argument("--component", required=True, help="e.g., n-pentane, n-heptane, methane, benzene ... (see CRIT_DB/ANTOINE_BAR_K)")
    ap.add_argument("--T", type=float, required=True, help="Temperature [K]")
    ap.add_argument("--P", type=float, required=True, help="Pressure [bar]")
    ap.add_argument("--method", choices=["raoult", "wilson", "depriester_mcwilliams"], default="wilson")
    ap.add_argument("--coeffs", default=None, help="Path to McWilliams coefficients JSON (if method=depriester_mcwilliams).")
    args = ap.parse_args()
    K = k_value(args.component, args.T, args.P, method=args.method, coeffs_path=args.coeffs)
    print(f"K({args.component}, T={args.T} K, P={args.P} bar, method={args.method}) = {K:.6g}")

if __name__ == "__main__":
    main()

# K-values from NIST-style Antoine (bar, K)
# log10(P_bar) = A - B / (T_K + C)
# -> P_bar = 10**(A - B/(T_K + C))

from typing import Dict, Tuple

# Paste your NIST Antoine parameters here:
# Each component: (A, B, C, Tmin_K, Tmax_K)
# NOTE: Use the *bar/K* set from the NIST page you’re using.
NIST_ANTOINE: Dict[str, Tuple[float, float, float, float, float]] = {
    # EXAMPLES (placeholders — replace with values copied from your NIST pages):
    "n-butane":       (3.85002, 909.65, -36.146, 195.11, 272.81),
    "1-butene":       (4.24696, 1099.207, -8.256, 195.7, 269.4),
    "i-butane":       (3.94417, 912.141 , -29.808, 188.06, 261.54),
    "cis-2-butene":   (3.98744, 957.06, -36.504, 203.06, 295.91),
    "trans-2-butene": (4.0436, 982.166, -30.775, 201.7, 274.13)
}

def antoine_psat_bar(component: str, T_K: float) -> float:
    """Return saturation pressure in bar using NIST Antoine (bar, K)."""
    try:
        A, B, C, Tmin, Tmax = NIST_ANTOINE[component]
    except KeyError:
        raise KeyError(f"Component '{component}' not found in NIST_ANTOINE.")
    if not (Tmin <= T_K <= Tmax):
        # Not fatal, but warn to keep you honest about ranges
        print(f"⚠️  {component}: T={T_K:.2f} K is outside recommended range [{Tmin:.2f}, {Tmax:.2f}] K.")
    return 10.0 ** (A - B / (T_K + C))

def K_value(component: str, T_K: float, P_column_bar: float) -> float:
    """Equilibrium ratio K = Psat / P (ideal)."""
    Psat = antoine_psat_bar(component, T_K)
    return Psat / P_column_bar

def print_table(components, T_K: float, P_bar: float):
    print(f"At T = {T_K:.2f} K, P = {P_bar:.3f} bar")
    print(f"{'Component':16s} {'Psat [bar]':>12s} {'K = Psat/P':>12s}")
    for comp in components:
        Psat = antoine_psat_bar(comp, T_K)
        K = Psat / P_bar
        print(f"{comp:16s} {Psat:12.5f} {K:12.5f}")
    # Volatility ranking
    ranked = sorted(components, key=lambda c: K_value(c, T_K, P_bar), reverse=True)
    print("\nVolatility ranking (highest K first):")
    for i, c in enumerate(ranked, 1):
        print(f"{i:2d}. {c}")

if __name__ == "__main__":
    # --- USER INPUTS ---
    T_K = 250     # e.g. 70 °C = 343.15 K
    P_bar = 1.0     # column pressure in bar
    components = [
        # fill to match what you added in NIST_ANTOINE
        "i-butane", "n-butane", "1-butene", "trans-2-butene", "cis-2-butene"
    ]

    if not components:
        print("Add your components to 'components' and paste NIST Antoine (bar,K) into NIST_ANTOINE above.")
    else:
        print_table(components, T_K, P_bar)

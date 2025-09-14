import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# Original code this is built on: https://colab.research.google.com/drive/1MJpqPV9T2hf0-RuOOeoFKb0ZXvLABr4y?usp=sharing McCabe-Thiele distillation code

# ---------- Equilibrium (constant alpha) ----------
def calculate_equilibrium_curve(alpha):
    # Build the equilibrium y–x curve assuming constant relative volatility (α).
    # y = α x / [1 + (α − 1) x]  (Raoult/ideal, constant-α shortcut)
    x = np.linspace(0, 1, 101)
    y = (alpha * x) / (1 + (alpha - 1) * x)
    return x, y

def solve_equilibrium_x_from_y(y, alpha):
    """Invert y = alpha x / (1 + (alpha-1) x) for x."""
    # We need x at a given y on the equilibrium curve (horizontal step to VLE).
    # Solve the scalar equation with fsolve; clamp to [0,1] for safety.
    def f(x): return (alpha * x) / (1 + (alpha - 1) * x) - y
    x_guess = y / max(alpha, 1.0000001)  # reasonable initial guess
    x_sol = float(fsolve(f, x_guess)[0])
    return float(np.clip(x_sol, 0.0, 1.0))

# ---------- Lines ----------
def calculate_feed_line(xF, q):
    # q-line (feed condition line) in McCabe–Thiele:
    # q=1 (saturated liquid) → vertical at x = xF
    # otherwise y = (q/(q-1)) x − xF/(q-1)
    if q == 1:
        return np.array([xF, xF]), np.array([0, 1])
    slope = q / (q - 1)
    intercept = -xF / (q - 1)
    x = np.array([0, 1])
    y = slope * x + intercept
    return x, y

def calculate_rectifying_line(D_comp, R, condenser_type):
    """
    Rectifying operating line (above the feed) under CMO:
      y = (R/(R+1)) x + (D_comp)/(R+1)
    - For a total condenser or partial_liquid condenser, D_comp = xD (liquid distillate).
    - For a partial_vapour condenser, D_comp = yD (vapour distillate).
    Note: drawing over [0,1] keeps the plot clean; slope/intercept returned for stepping.
    """
    slope = R / (R + 1)
    intercept = D_comp / (R + 1)   # xD or yD depending on condenser
    x = np.array([0.0, 1.0])       # draw full segment (cleaner)
    y = slope * x + intercept
    return x, y, slope, intercept

def calculate_stripping_line(xB, xD_or_yD, xF, R, q, rect_slope, rect_intercept):
    # Find intersection M between rectifying line and q-line.
    # For q=1 (vertical), M is simply at x = xF on the rectifying OL.
    # For general q, intersect y = m_rect x + b_rect with y = (q/(q-1)) x − xF/(q-1).
    if q == 1:
        x_intersect = xF
        y_intersect = rect_slope * xF + rect_intercept
    else:
        feed_slope = q / (q - 1)
        feed_intercept = -xF / (q - 1)
        x_intersect = (feed_intercept - rect_intercept) / (rect_slope - feed_slope)
        y_intersect = rect_slope * x_intersect + rect_intercept

    # Stripping operating line (below the feed) passes through:
    #   - the intersection M(x_intersect, y_intersect)
    #   - the bottoms equilibrium point (xB, xB) because total/partial reboiler returns liquid at xB
    slope = (y_intersect - xB) / (x_intersect - xB)
    intercept = xB - slope * xB
    x = np.array([xB, x_intersect])
    y = slope * x + intercept
    return (x, y, x_intersect, y_intersect, slope, intercept)

# ---------- Stages ----------
def calculate_stages(xB, D_comp, alpha, R, xF, q,
                     condenser_type, reboiler_type, stripping_line, rect_slope, rect_intercept):
    """
    McCabe–Thiele stepping and stage counting.
    We count half-steps (horizontal to VLE + vertical to OL) and divide by 2 for stages.
    Edge rules handled:
      - Condenser type:
          * total:   start at (xD, xD)  → no condenser stage counted.
          * partial_liquid: start at (xD, y*(xD)) then vertical to rectifying OL (counts a stage).
          * partial_vapour: start at (x*(yD), yD) then vertical to rectifying OL (counts a stage).
      - Reboiler type (total/partial): in either case we add the final vertical at x=xB to (xB,xB),
        which counts the reboiler stage.
    """
    stages = []
    stage_points_x, stage_points_y = [], []
    stage_halfsteps = 0  # each horizontal/vertical is one half-stage

    # Feed intersection used to switch from rectifying to stripping line
    x_intersect, y_intersect = stripping_line[2], stripping_line[3]

    # ---- START at the correct point depending on condenser ----
    if condenser_type == "total":
        # Total condenser → distillate is liquid at (xD, xD); no extra stage at the top.
        x = D_comp     # xD
        y = D_comp     # xD
        stage_points_x += [x]; stage_points_y += [y]
        stages.append({"section": "Start (total condenser)", "x": x, "y": y})

    elif condenser_type == "partial_liquid":
        # Liquid product from a partial condenser:
        #   top-tray liquid at xD is in equilibrium with vapour y*(xD).
        #   Vertical to the rectifying line is the condenser stage.
        x = D_comp                           # xD
        y = (alpha * x) / (1 + (alpha - 1) * x)  # y*(xD)
        stage_points_x += [x]; stage_points_y += [y]
        stages.append({"section": "Condenser eq (vapour)", "x": x, "y": y})

        y_rect = rect_slope * x + rect_intercept  # vertical to OL (condenser stage)
        stage_points_x += [x]; stage_points_y += [y_rect]
        stages.append({"section": "Condenser (vertical to OL)", "x": x, "y": y_rect})
        stage_halfsteps += 2
        y = y_rect

    elif condenser_type == "partial_vapour":
        # Vapour product from a partial condenser:
        #   top-tray vapour yD is in equilibrium with liquid x*(yD).
        #   Vertical to the rectifying line is the condenser stage.
        y = D_comp
        x = solve_equilibrium_x_from_y(y, alpha)  # x* in eq with yD
        stage_points_x += [x]; stage_points_y += [y]
        stages.append({"section": "Condenser eq (liquid)", "x": x, "y": y})

        y_rect = rect_slope * x + rect_intercept
        stage_points_x += [x]; stage_points_y += [y_rect]
        stages.append({"section": "Condenser (vertical to OL)", "x": x, "y": y_rect})
        stage_halfsteps += 2
        y = y_rect
    else:
        # Guard for invalid input.
        raise ValueError("condenser_type must be one of: total, partial_liquid, partial_vapour")

    is_rectifying = True  # we start above the feed

    # ---- Tray-to-tray stepping loop ----
    while True:
        # Horizontal step to the equilibrium curve: (x,y) → (x_on_curve, y)
        x_on_curve = solve_equilibrium_x_from_y(y, alpha)
        stage_points_x.append(x_on_curve); stage_points_y.append(y)
        stages.append({"section": "Horizontal to VLE", "x": x_on_curve, "y": y})
        stage_halfsteps += 1

        # When the horizontal crosses the feed intersection x-value, switch to stripping line
        if is_rectifying and x_on_curve <= x_intersect + 1e-9:
            is_rectifying = False

        # Vertical step to the operating line (rectifying above feed, stripping below)
        if is_rectifying:
            y_on_op = rect_slope * x_on_curve + rect_intercept
        else:
            strip_slope, strip_intercept = stripping_line[4], stripping_line[5]
            y_on_op = strip_slope * x_on_curve + strip_intercept

        stage_points_x.append(x_on_curve); stage_points_y.append(y_on_op)
        stages.append({"section": "Vertical to OL", "x": x_on_curve, "y": y_on_op})
        stage_halfsteps += 1

        x, y = x_on_curve, y_on_op  # update current point

        # Termination: when liquid composition reaches xB (or slightly lower due to tolerance)
        if x <= xB + 1e-3:
            # Reboiler stage: final vertical from the operating line down to (xB, xB) on 45° line.
            stage_points_x.append(xB); stage_points_y.append(xB)
            stages.append({"section": "Reboiler (vertical to 45°)", "x": xB, "y": xB})
            stage_halfsteps += 2
            break

        # Safety to avoid infinite loops if inputs are inconsistent
        if stage_halfsteps > 400:
            break

    # Convert half-steps to stages (no flooring → keeps fractional stage)
    N_theoretical = stage_halfsteps / 2.0
    return {
        "stages": stages,
        "stage_points": (stage_points_x, stage_points_y),
        "actual_stage_count": N_theoretical,
        "x_intersect": x_intersect,
        "y_intersect": y_intersect
    }

# ---------- Plot ----------
def plot_diagram(eq_points, feed_line, rectifying_line, stripping_line, stage_points):
    # Assemble the McCabe–Thiele chart: 45° line, equilibrium curve, q-line,
    # rectifying/stripping operating lines, and the stepped stage path.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='45° Line',
                             line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=eq_points[0], y=eq_points[1], mode='lines',
                             name='Equilibrium Curve', line=dict(color='#1E90FF', width=3)))
    fig.add_trace(go.Scatter(x=feed_line[0], y=feed_line[1], mode='lines',
                             name='Feed Line', line=dict(color='#32CD32', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=rectifying_line[0], y=rectifying_line[1], mode='lines',
                             name='Rectifying Line', line=dict(color='#FF4500', width=3)))
    fig.add_trace(go.Scatter(x=stripping_line[0], y=stripping_line[1], mode='lines',
                             name='Stripping Line', line=dict(color='#8A2BE2', width=3)))
    fig.add_trace(go.Scatter(x=stage_points[0], y=stage_points[1], mode='lines+markers',
                             name='Stages', line=dict(color='orange', width=3),
                             marker=dict(size=6, color='orange')))
    fig.update_layout(
        title=dict(text='McCabe-Thiele Diagram', font=dict(size=16), y=0.95),
        xaxis=dict(title='x (Liquid Composition)', range=[0,1], dtick=0.2),
        yaxis=dict(title='y (Vapor Composition)', range=[0,1], dtick=0.2),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(t=80, b=80, l=80, r=40), width=700, height=500
    )
    fig.show()

def display_results(alpha, xF, D_comp, xB, q, R, stages_result, condenser_type, reboiler_type, rect_slope, rect_intercept):
    # Console summary: stage count (with condenser/reboiler types), key line equations, and feed intersection.
    N = stages_result['actual_stage_count']
    print(f"Number of Theoretical Stages (incl. {condenser_type} condenser & {reboiler_type} reboiler): {N:.2f}  (ceil → {int(np.ceil(N))})")
    print(f"Operating Reflux Ratio: {R:.3f}")
    print(f"Feed/OL Intersection: ({stages_result['x_intersect']:.4f}, {stages_result['y_intersect']:.4f})")
    print(f"Rectifying Line: y = {rect_slope:.4f} x + {rect_intercept:.4f}  "
          f"[{'xD' if condenser_type!='partial_vapour' else 'yD'} = {D_comp}]")
    print(f"Stripping Line:  y = {stages_result['stripping_line'][4]:.4f} x + {stages_result['stripping_line'][5]:.4f}")
    print(f"Feed Line: {'x = ' + str(xF) if q == 1 else f'y = {(q/(q-1)):.4f} x - {(xF/(q-1)):.4f}'}")

# ---------- GUI ----------
def get_user_inputs_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    # Defaults left as 0s as requested (you will enter values in the popup).
    defaults = {"alpha": 0, "xF": 0, "D_comp": 0, "xB": 0, "q": 0, "R": 0,
                "condenser": "total", "reboiler": "total"}
    result = {}

    def submit():
        try:
            # Read user inputs from the form
            alpha = float(alpha_var.get())
            xF = float(xF_var.get())
            D_comp = float(D_var.get())  # xD or yD depending on condenser choice
            xB = float(xB_var.get())
            q = float(q_var.get())
            R = float(R_var.get())
            condenser = condenser_var.get()
            reboiler = reboiler_var.get()

            # Minimal validation to avoid impossible diagrams
            if alpha <= 1.0: raise ValueError("α must be > 1.")
            for name, val in [("xF", xF), ("D_comp", D_comp), ("xB", xB)]:
                if not (0.0 < val < 1.0): raise ValueError(f"{name} must be in (0,1).")
            if not (xB < xF): raise ValueError("Require xB < xF. (xF < xD or yD is your responsibility.)")
            if q < 0.0: raise ValueError("q must be ≥ 0.")
            if R < 0.0: raise ValueError("R must be ≥ 0.")
            if condenser not in ("total","partial_liquid","partial_vapour"): raise ValueError("Invalid condenser type.")
            if reboiler not in ("total","partial"): raise ValueError("Invalid reboiler type.")

            # If user typed 0.99999999 etc., snap to exactly 1 for a vertical q-line.
            if abs(q - 1.0) < 1e-9: q = 1.0

            # Store and close popup
            result.update(dict(alpha=alpha, xF=xF, D_comp=D_comp, xB=xB, q=q, R=R,
                               condenser=condenser, reboiler=reboiler))
            root.destroy()
        except ValueError as e:
            # Show a dialog with the validation message
            messagebox.showerror("Invalid input", str(e))

    # --- Tkinter widgets / layout ---
    root = tk.Tk()
    root.title("McCabe–Thiele Inputs")
    root.resizable(False, False)
    pad = {"padx": 10, "pady": 6}

    # StringVar bindings for the entries and dropdowns
    alpha_var = tk.StringVar(value=str(defaults["alpha"]))
    xF_var    = tk.StringVar(value=str(defaults["xF"]))
    D_var     = tk.StringVar(value=str(defaults["D_comp"]))
    xB_var    = tk.StringVar(value=str(defaults["xB"]))
    q_var     = tk.StringVar(value=str(defaults["q"]))
    R_var     = tk.StringVar(value=str(defaults["R"]))
    condenser_var = tk.StringVar(value=defaults["condenser"])
    reboiler_var  = tk.StringVar(value=defaults["reboiler"])

    frm = ttk.Frame(root); frm.grid(row=0, column=0, sticky="nsew", **pad)

    # Dropdowns: types of condenser and reboiler (affect starting/ending steps)
    ttk.Label(frm, text="Condenser type").grid(row=0, column=0, sticky="w", **pad)
    ttk.Combobox(frm, textvariable=condenser_var, values=["total","partial_liquid","partial_vapour"],
                 state="readonly", width=16).grid(row=0, column=1, **pad)

    ttk.Label(frm, text="Reboiler type").grid(row=1, column=0, sticky="w", **pad)
    ttk.Combobox(frm, textvariable=reboiler_var, values=["total","partial"],
                 state="readonly", width=16).grid(row=1, column=1, **pad)

    # Numeric entries
    ttk.Label(frm, text="Relative volatility α (>1)").grid(row=2, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=alpha_var, width=18).grid(row=2, column=1, **pad)

    ttk.Label(frm, text="Feed composition xF (0–1)").grid(row=3, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=xF_var, width=18).grid(row=3, column=1, **pad)

    ttk.Label(frm, text="Distillate composition (xD if total/partial_liquid, yD if partial_vapour)").grid(row=4, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=D_var, width=18).grid(row=4, column=1, **pad)

    ttk.Label(frm, text="Bottoms composition xB (0–1)").grid(row=5, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=xB_var, width=18).grid(row=5, column=1, **pad)

    ttk.Label(frm, text="Feed quality q (0=dew vap., 1=bubble liq.)").grid(row=6, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=q_var, width=18).grid(row=6, column=1, **pad)

    ttk.Label(frm, text="Reflux ratio R (≥0)").grid(row=7, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=R_var, width=18).grid(row=7, column=1, **pad)

    # Buttons and bindings
    btns = ttk.Frame(frm); btns.grid(row=8, column=0, columnspan=2, sticky="e", **pad)
    ttk.Button(btns, text="Cancel", command=root.destroy).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="OK", command=submit).grid(row=0, column=1, padx=6)
    root.bind("<Return>", lambda e: submit())
    root.mainloop()

    if not result:
        # User closed the window without valid inputs
        raise SystemExit("Input window closed.")
    return (result["alpha"], result["xF"], result["D_comp"], result["xB"], result["q"], result["R"],
            result["condenser"], result["reboiler"])

# ---------- Main ----------
def main():
    # Collect inputs from the popup; D_comp will be xD (liquid D) or yD (vapour D) based on condenser type.
    alpha, xF, D_comp, xB, q, R, condenser_type, reboiler_type = get_user_inputs_gui()

    # Build the equilibrium curve and all lines (q-line, operating lines)
    eq_x, eq_y = calculate_equilibrium_curve(alpha)
    feed_line = calculate_feed_line(xF, q)
    rect_line_x, rect_line_y, rect_slope, rect_intercept = calculate_rectifying_line(D_comp, R, condenser_type)
    stripping_line = calculate_stripping_line(xB, D_comp, xF, R, q, rect_slope, rect_intercept)

    # Step off stages and count them with the chosen condenser/reboiler logic
    stages_result = calculate_stages(
        xB, D_comp, alpha, R, xF, q, condenser_type, reboiler_type,
        stripping_line, rect_slope, rect_intercept
    )
    stages_result["stripping_line"] = stripping_line  # for printing coefficients

    # Plot the full McCabe–Thiele diagram and print a tidy summary
    plot_diagram((eq_x, eq_y), feed_line,
                 (rect_line_x, rect_line_y), stripping_line,
                 stage_points=stages_result["stage_points"])

    display_results(alpha, xF, D_comp, xB, q, R, stages_result,
                    condenser_type, reboiler_type, rect_slope, rect_intercept)

if __name__ == "__main__":
    main()

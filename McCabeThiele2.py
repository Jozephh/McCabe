import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# Where I found original code https://colab.research.google.com/drive/1MJpqPV9T2hf0-RuOOeoFKb0ZXvLABr4y?usp=sharing McCabe-Thiele distillation code

def calculate_equilibrium_curve(alpha):
    """Calculate equilibrium curve points based on relative volatility."""
    x = np.linspace(0, 1, 101)
    y = (alpha * x) / (1 + (alpha - 1) * x)
    return x, y

def calculate_feed_line(xF, q):
    """Calculate the feed line based on feed composition and quality."""
    if q == 1:
        return np.array([xF, xF]), np.array([0, 1])
    else:
        slope = q / (q - 1)
        intercept = -xF / (q - 1)
        x = np.array([0, 1])
        y = slope * x + intercept
        return x, y

def calculate_rectifying_line(xD, R):
    """Calculate the rectifying section operating line."""
    slope = R / (R + 1)
    intercept = xD / (R + 1)
    x = np.array([0, xD, xD])
    y = np.array([intercept, slope * xD + intercept, xD])
    return x, y

def calculate_stripping_line(xB, xD, xF, R, q):
    """Calculate the stripping section operating line."""
    rect_slope = R / (R + 1)
    rect_intercept = xD / (R + 1)

    if q == 1:
        x_intersect = xF
        y_intersect = rect_slope * xF + rect_intercept
    else:
        feed_slope = q / (q - 1)
        feed_intercept = -xF / (q - 1)
        x_intersect = (feed_intercept - rect_intercept) / (rect_slope - feed_slope)
        y_intersect = rect_slope * x_intersect + rect_intercept

    slope = (y_intersect - xB) / (x_intersect - xB)
    intercept = xB - slope * xB
    x = np.array([xB, xB, x_intersect])
    y = np.array([xB, xB, y_intersect])

    return x, y, x_intersect, y_intersect, slope, intercept

def solve_equilibrium_x(y, alpha):
    """Solve for x given y on the equilibrium curve."""
    def func(x):
        return (alpha * x) / (1 + (alpha - 1) * x) - y
    x_guess = y / alpha
    x_solution = fsolve(func, x_guess)[0]
    return max(0, min(1, x_solution))

def calculate_stages(xB, xD, alpha, R, xF, q, stripping_line):
    """Calculate the number of theoretical stages and their coordinates."""
    stages = []
    stage_points_x = [xD]
    stage_points_y = [xD]
    x = xD
    y = xD
    stage_count = 0
    is_rectifying = True

    rect_slope = R / (R + 1)
    rect_intercept = xD / (R + 1)

    if q == 1:
        x_intersect = xF
        y_intersect = rect_slope * xF + rect_intercept
    else:
        feed_slope = q / (q - 1)
        feed_intercept = -xF / (q - 1)
        x_intersect = (feed_intercept - rect_intercept) / (rect_slope - feed_slope)
        y_intersect = rect_slope * x_intersect + rect_intercept

    stages.append({"stage_number": stage_count, "x": x, "y": y, "section": "Start"})

    while x > xB + 0.001 and stage_count < 100:
        # Step to equilibrium curve (horizontal)
        x_on_curve = solve_equilibrium_x(y, alpha)
        stage_count += 1
        stages.append({"stage_number": stage_count, "x": x_on_curve, "y": y, "section": "Horizontal"})
        stage_points_x.append(x_on_curve)
        stage_points_y.append(y)

        # Step to operating line (vertical)
        if is_rectifying and x_on_curve > x_intersect:
            y_on_op_line = rect_slope * x_on_curve + rect_intercept
        else:
            is_rectifying = False
            strip_slope = stripping_line[4]
            strip_intercept = stripping_line[5]
            y_on_op_line = strip_slope * x_on_curve + strip_intercept

        stage_count += 1
        stages.append({"stage_number": stage_count, "x": x_on_curve, "y": y_on_op_line, "section": "Vertical"})
        stage_points_x.append(x_on_curve)
        stage_points_y.append(y_on_op_line)

        x = x_on_curve
        y = y_on_op_line

        # If x is close to xB, add final point and break
        if x <= xB + 0.001:
            if x > xB:
                x_on_curve = xB
                y_on_op_line = xB
                stage_count += 1
                stages.append({"stage_number": stage_count, "x": x_on_curve, "y": y_on_op_line, "section": "Final"})
                stage_points_x.append(x_on_curve)
                stage_points_y.append(y_on_op_line)
            break

    actual_stage_count = int(stage_count / 2)
    return {
        "stages": stages,
        "stage_points": (stage_points_x, stage_points_y),
        "actual_stage_count": actual_stage_count,
        "x_intersect": x_intersect,
        "y_intersect": y_intersect
    }

def plot_diagram(eq_points, feed_line, rectifying_line, stripping_line, stage_points):
    """Plot the McCabe-Thiele diagram using Plotly."""
    fig = go.Figure()

    # 45-degree line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='45° Line',
        line=dict(color='black', width=3)
    ))

    # Equilibrium curve
    fig.add_trace(go.Scatter(
        x=eq_points[0], y=eq_points[1],
        mode='lines',
        name='Equilibrium Curve',
        line=dict(color='#1E90FF', width=3)
    ))

    # Feed line
    fig.add_trace(go.Scatter(
        x=feed_line[0], y=feed_line[1],
        mode='lines',
        name='Feed Line',
        line=dict(color='#32CD32', width=3, dash='dash')
    ))

    # Rectifying line
    fig.add_trace(go.Scatter(
        x=rectifying_line[0], y=rectifying_line[1],
        mode='lines',
        name='Rectifying Line',
        line=dict(color='#FF4500', width=3)
    ))

    # Stripping line
    fig.add_trace(go.Scatter(
        x=stripping_line[0], y=stripping_line[1],
        mode='lines',
        name='Stripping Line',
        line=dict(color='#8A2BE2', width=3)
    ))

    # Stages
    fig.add_trace(go.Scatter(
        x=stage_points[0], y=stage_points[1],
        mode='lines+markers',
        name='Stages',
        line=dict(color='orange', width=3),
        marker=dict(size=6, color='orange')
    ))

    # Layout
    fig.update_layout(
        title=dict(text='McCabe-Thiele Diagram', font=dict(size=16), y=0.95),
        xaxis=dict(
            title=dict(text='x (Liquid Composition)', font=dict(size=12), standoff=10),
            range=[0, 1],
            dtick=0.2
        ),
        yaxis=dict(
            title=dict(text='y (Vapor Composition)', font=dict(size=12), standoff=10),
            range=[0, 1],
            dtick=0.2
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(t=80, b=80, l=80, r=40),
        showlegend=True,
        width=700,
        height=500
    )

    fig.show()

def display_results(alpha, xF, xD, xB, q, R, stages_result):
    """Print calculation results to console."""
    print(f"Number of Theoretical Stages: {stages_result['actual_stage_count']}")
    print(f"Operating Reflux Ratio: {R:.2f}")
    print(f"Feed Line and Operating Line Intersection: ({stages_result['x_intersect']:.4f}, {stages_result['y_intersect']:.4f})")
    print(f"Rectifying Line Equation: y = {(R/(R+1)):.4f}x + {(xD/(R+1)):.4f}")
    print(f"Stripping Line Equation: y = {stages_result['stripping_line'][4]:.4f}x + {stages_result['stripping_line'][5]:.4f}")
    print(f"Feed Line Equation: {'x = ' + str(xF) if q == 1 else f'y = {(q/(q-1)):.4f}x - {(xF/(q-1)):.4f}'}")

# -------- NEW: Popup form (Tkinter) --------
def get_user_inputs_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    defaults = {"alpha": 3.166, "xF": 0.46, "xD": 0.95, "xB": 0.18, "q": 1.0, "R": 2.0}
    result = {}

    def submit():
        try:
            alpha = float(alpha_var.get())
            xF = float(xF_var.get())
            xD = float(xD_var.get())
            xB = float(xB_var.get())
            q = float(q_var.get())
            R = float(R_var.get())

            # Validations (same logic as before)
            if alpha <= 1.0:
                raise ValueError("Relative volatility must be > 1.")
            for name, val in [("xF", xF), ("xD", xD), ("xB", xB)]:
                if not (0.0 < val < 1.0):
                    raise ValueError(f"{name} must be between 0 and 1 (exclusive).")
            if not (xB < xF < xD):
                raise ValueError("Composition order must be xB < xF < xD.")
            if q < 0.0:
                raise ValueError("Feed quality q must be ≥ 0.")
            if R < 0.0:
                raise ValueError("Reflux ratio R must be ≥ 0.")

            # Snap q ~ 1 exactly to 1 for a vertical q-line
            if abs(q - 1.0) < 1e-9:
                q = 1.0

            result.update(dict(alpha=alpha, xF=xF, xD=xD, xB=xB, q=q, R=R))
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))

    def cancel():
        root.destroy()

    root = tk.Tk()
    root.title("McCabe–Thiele Inputs")
    root.resizable(False, False)

    # A little padding
    pad = {"padx": 10, "pady": 6}

    # Variables with defaults
    alpha_var = tk.StringVar(value=str(defaults["alpha"]))
    xF_var    = tk.StringVar(value=str(defaults["xF"]))
    xD_var    = tk.StringVar(value=str(defaults["xD"]))
    xB_var    = tk.StringVar(value=str(defaults["xB"]))
    q_var     = tk.StringVar(value=str(defaults["q"]))
    R_var     = tk.StringVar(value=str(defaults["R"]))

    frm = ttk.Frame(root)
    frm.grid(row=0, column=0, sticky="nsew", **pad)

    # Labels + entries (grid)
    rows = [
        ("Relative volatility α (>1)", alpha_var),
        ("Feed composition xF (0–1)", xF_var),
        ("Distillate composition xD (0–1)", xD_var),
        ("Bottoms composition xB (0–1)", xB_var),
        ("Feed quality q (0=dew vap., 1=bubble liq.)", q_var),
        ("Reflux ratio R (≥0)", R_var),
    ]
    for i, (label, var) in enumerate(rows):
        ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=var, width=18).grid(row=i, column=1, **pad)

    # Buttons
    btns = ttk.Frame(frm)
    btns.grid(row=len(rows), column=0, columnspan=2, sticky="e", **pad)
    ttk.Button(btns, text="Cancel", command=cancel).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="OK", command=submit).grid(row=0, column=1, padx=6)

    # Make Enter key submit
    root.bind("<Return>", lambda e: submit())

    root.mainloop()

    if not result:
        raise SystemExit("Input window closed.")

    return result["alpha"], result["xF"], result["xD"], result["xB"], result["q"], result["R"]

def main():
    # ---- Get inputs from popup instead of terminal ----
    alpha, xF, xD, xB, q, R = get_user_inputs_gui()

    # Calculate components
    eq_points = calculate_equilibrium_curve(alpha)
    feed_line = calculate_feed_line(xF, q)
    rectifying_line = calculate_rectifying_line(xD, R)
    stripping_line = calculate_stripping_line(xB, xD, xF, R, q)
    stages_result = calculate_stages(xB, xD, alpha, R, xF, q, stripping_line)
    stages_result["stripping_line"] = stripping_line

    # Plot and display results
    plot_diagram(eq_points, feed_line, rectifying_line, stripping_line, stage_points=stages_result["stage_points"])
    display_results(alpha, xF, xD, xB, q, R, stages_result)

# Run the main function
if __name__ == "__main__":
    main()

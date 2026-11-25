"""
================================================================================
AI39 | AI Network Anomaly Detector
MATH 301 - Numerical Computations Laboratory
Main Application Server (Flask)

Features:
  - Single app.py backend with clean modular endpoints.
  - Zero database dependency; in-memory circular state + CSV export.
  - Integration with pure-Python numerical_engine.py.
  - Robust cold-start handling (t < 3, t < N) & zero-division safety.
================================================================================
"""

import io
import csv
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response

from numerical_engine import (
    FiniteDifferenceEngine,
    LeastSquaresEngine,
    AnomalyEvaluator,
    ConfusionMatrixTracker,
    NetworkTrafficSimulator,
    run_50_sample_benchmark
)

app = Flask(__name__)

# ------------------------------------------------------------------------------
# IN-MEMORY SYSTEM STATE
# ------------------------------------------------------------------------------
class SystemState:
    def __init__(self):
        # Parameters (user adjustable)
        self.tau_d: float = 35.0      # Derivative threshold (pkt/s^2)
        self.tau_r: float = 28.0      # Residual threshold (pkt/s)
        self.window_n: int = 20       # Sliding window size N
        self.base_noise: float = 3.5  # Noise level (+/- pkt/s)

        # Simulation & Math Engines
        self.simulator = NetworkTrafficSimulator(base_noise=self.base_noise)
        self.confusion_matrix = ConfusionMatrixTracker()

        # Telemetry History & Sliding Window Memory
        self.history = []               # All recorded observations
        self.raw_y = []                 # Historical raw y values for finite differences
        self.clean_window_t = []        # Anti-corruption baseline window t
        self.clean_window_y = []        # Anti-corruption baseline window y

    def reset(self):
        self.simulator.reset()
        self.confusion_matrix.reset()
        self.history.clear()
        self.raw_y.clear()
        self.clean_window_t.clear()
        self.clean_window_y.clear()


state = SystemState()


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    """Serves the single-page telemetry and control dashboard."""
    return render_template("index.html")


@app.route("/api/tick", methods=["GET"])
def tick():
    """
    Advances simulation by 1 second (Delta_t = 1.0 s) and computes:
      1. Numerical Derivative (1st-order and 3-point backward difference)
      2. Least Squares Linear Fit over clean window of size N
      3. Anomaly Evaluation & Online Confusion Matrix
    """
    # Advance simulator
    t, y_curr, ground_truth, injection_desc = state.simulator.next_sample()
    state.raw_y.append(y_curr)
    num_samples = len(state.raw_y)
    dt = 1.0

    # --------------------------------------------------------------------------
    # 1. Method A: Numerical Differentiation (Finite Differences)
    # --------------------------------------------------------------------------
    if num_samples >= 2:
        y_prev = state.raw_y[-2]
        d_back1 = FiniteDifferenceEngine.backward_first_order(y_curr, y_prev, dt)
        formula_deriv = f"({y_curr:.1f} - {y_prev:.1f}) / {dt:.1f} = {d_back1:+.2f} pkt/s²"
    else:
        # Initial cold-start t = 1
        d_back1 = 0.0
        formula_deriv = "t=1: Initializing baseline (y' = 0.00)"

    if num_samples >= 3:
        y_prev = state.raw_y[-2]
        y_prev2 = state.raw_y[-3]
        d_back3 = FiniteDifferenceEngine.backward_three_point(y_curr, y_prev, y_prev2, dt)
    else:
        d_back3 = d_back1

    # --------------------------------------------------------------------------
    # 2. Method B: Least Squares Linear Curve Fitting
    # --------------------------------------------------------------------------
    if len(state.clean_window_t) >= 3:
        # Slice the last N clean points
        sub_t = state.clean_window_t[-state.window_n:]
        sub_y = state.clean_window_y[-state.window_n:]
        a_0, a_1, sigma_est = LeastSquaresEngine.fit_linear(sub_t, sub_y)
        y_hat = LeastSquaresEngine.predict(a_0, a_1, float(t))
        sign_str = "+" if a_1 >= 0 else "-"
        formula_ls = f"ŷ(t) = {a_0:.2f} {sign_str} {abs(a_1):.3f}·t  (N={len(sub_t)}, σ={sigma_est:.2f})"
    else:
        # Cold start: when points < 3, baseline equals current observation
        a_0 = y_curr
        a_1 = 0.0
        sigma_est = 0.0
        y_hat = y_curr
        formula_ls = f"ŷ(t) ≈ {y_curr:.1f} (Cold-start buffer < 3)"

    # --------------------------------------------------------------------------
    # 3. Anomaly Evaluation & Practical Decision Rule
    # --------------------------------------------------------------------------
    eval_res = AnomalyEvaluator.evaluate(
        y_curr=y_curr,
        y_hat=y_hat,
        y_prime_1st=d_back1,
        tau_d=state.tau_d,
        tau_r=state.tau_r
    )
    detected = eval_res["detected_anomaly"]

    # Update live confusion matrix
    state.confusion_matrix.update(detected=detected, ground_truth=ground_truth)

    # --------------------------------------------------------------------------
    # 4. Anti-Corruption Window Maintenance
    # --------------------------------------------------------------------------
    # If the observation is NOT an anomaly, add the true observation.
    # If it is an anomaly (DDoS spike, glitch, or severe outage), clamp to y_hat
    # so that malicious traffic does not poison the Least Squares regression!
    if not detected and not ground_truth:
        state.clean_window_t.append(float(t))
        state.clean_window_y.append(y_curr)
    else:
        state.clean_window_t.append(float(t))
        state.clean_window_y.append(round(y_hat, 2))

    # Keep memory bounded (last 1000 items)
    if len(state.clean_window_t) > 1000:
        state.clean_window_t.pop(0)
        state.clean_window_y.pop(0)

    # Build record
    record = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "t": t,
        "y": round(y_curr, 2),
        "y_hat": round(y_hat, 2),
        "a_0": round(a_0, 3),
        "a_1": round(a_1, 4),
        "sigma_est": round(sigma_est, 2),
        "residual": eval_res["residual"],
        "d_back1": round(d_back1, 2),
        "d_back3": round(d_back3, 2),
        "formula_deriv": formula_deriv,
        "formula_ls": formula_ls,
        "status": eval_res["status"],
        "reason": eval_res["reason"],
        "detected_anomaly": detected,
        "ground_truth_anomaly": ground_truth,
        "injection_desc": injection_desc
    }

    state.history.append(record)
    if len(state.history) > 2000:
        state.history.pop(0)

    return jsonify({
        "current": record,
        "metrics": state.confusion_matrix.get_metrics(),
        "params": {
            "tau_d": state.tau_d,
            "tau_r": state.tau_r,
            "window_n": state.window_n,
            "base_noise": state.base_noise
        }
    })


@app.route("/api/inject", methods=["POST"])
def inject():
    """Triggers an anomaly mode (ddos, stealth, glitch, outage)."""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "ddos")
    state.simulator.trigger_injection(mode)
    return jsonify({
        "status": "success",
        "mode": mode,
        "message": f"Anomaly injection '{mode}' activated."
    })


@app.route("/api/params", methods=["POST"])
def update_params():
    """Updates user-adjustable numerical parameters."""
    data = request.get_json(silent=True) or {}
    if "tau_d" in data:
        state.tau_d = float(data["tau_d"])
    if "tau_r" in data:
        state.tau_r = float(data["tau_r"])
    if "window_n" in data:
        state.window_n = max(3, min(60, int(data["window_n"])))
    if "base_noise" in data:
        noise = max(0.0, min(20.0, float(data["base_noise"])))
        state.base_noise = noise
        state.simulator.base_noise = noise

    return jsonify({
        "status": "success",
        "params": {
            "tau_d": state.tau_d,
            "tau_r": state.tau_r,
            "window_n": state.window_n,
            "base_noise": state.base_noise
        }
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Resets simulator, telemetry history, and confusion matrix."""
    state.reset()
    return jsonify({
        "status": "success",
        "message": "Telemetry history, simulation clock, and confusion matrix reset."
    })


@app.route("/api/verify", methods=["GET"])
def verify_benchmark():
    """
    Runs the 50-sample held-out verification benchmark with analytical derivatives
    and returns full academic verification data.
    """
    res = run_50_sample_benchmark(
        tau_d=state.tau_d,
        tau_r=state.tau_r,
        window_n=state.window_n
    )
    return jsonify(res)


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns the last 50 historical points for chart initialization."""
    recent = state.history[-50:] if len(state.history) > 50 else state.history
    return jsonify({
        "history": recent,
        "metrics": state.confusion_matrix.get_metrics(),
        "params": {
            "tau_d": state.tau_d,
            "tau_r": state.tau_r,
            "window_n": state.window_n,
            "base_noise": state.base_noise
        }
    })


@app.route("/api/export_csv", methods=["GET"])
def export_csv():
    """Exports all recorded telemetry to a downloadable CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header columns according to MATH 301 specifications
    writer.writerow([
        "Timestamp",
        "Tick (t)",
        "Packet Rate y(t) [pkt/s]",
        "Least Squares Baseline y_hat(t) [pkt/s]",
        "Residual e_i [pkt/s]",
        "1st Order Derivative y' [pkt/s^2]",
        "3-Point Derivative y' [pkt/s^2]",
        "Fitted Slope a_1",
        "Fitted Intercept a_0",
        "Window RMSE sigma_est",
        "Decision Status",
        "Detected Anomaly",
        "Ground Truth Anomaly",
        "Traffic Description"
    ])

    for row in state.history:
        writer.writerow([
            row["timestamp"],
            row["t"],
            row["y"],
            row["y_hat"],
            row["residual"],
            row["d_back1"],
            row["d_back3"],
            row["a_1"],
            row["a_0"],
            row["sigma_est"],
            row["status"],
            "ANOMALY" if row["detected_anomaly"] else "NORMAL",
            "ANOMALY" if row["ground_truth_anomaly"] else "NORMAL",
            row["injection_desc"]
        ])

    csv_data = output.getvalue()
    filename = f"network_telemetry_MATH301_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("==================================================================")
    print("AI39 | AI Network Anomaly Detector - MATH 301")
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    print("==================================================================")
    app.run(host="127.0.0.1", port=5000, debug=False)

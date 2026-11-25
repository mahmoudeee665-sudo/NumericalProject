# AI39 | AI Network Anomaly Detector
### MATH 301 – Numerical Computations Laboratory Project

An elite, software-only cybersecurity and mathematical telemetry platform designed for **MATH 301 (Numerical Computations)**. The system performs real-time detection of volumetric denial-of-service (DDoS) floods, stealth linear ramps, network glitches, and link outages on discrete time-series telemetry ($y(t)$, packets/second, $\Delta t = 1.0\text{ s}$).

---

## 🎓 Academic Compliance & Rules

> **STRICT ACADEMIC INTEGRITY:**  
> In compliance with university guidelines for MATH 301, **zero third-party scientific or machine learning libraries** (`numpy.polyfit`, `scipy`, `scikit-learn`, `pandas`) are used for the numerical algorithms. All summation loops, normal equation solvers, and finite difference approximations are written in **pure Python** from scratch within `numerical_engine.py` for direct academic review and full credit.

---

## 📐 Mathematical Formulation

### 1. Method A: Numerical Differentiation (Finite Differences)
Detects sudden rate-of-change acceleration $\left|y'(t)\right| > \tau_d$ (default $\tau_d = 35.0\text{ pkt/s}^2$):

1. **First-Order Backward Finite Difference ($O(\Delta t)$ Truncation Error):**
   $$y'_{1st}(t_i) = \frac{y_i - y_{i-1}}{\Delta t}$$

2. **Three-Point Backward Difference ($O(\Delta t^2)$ Second-Order Accuracy):**
   $$y'_{3pt}(t_i) = \frac{3y_i - 4y_{i-1} + y_{i-2}}{2\Delta t}$$

3. **Central Finite Difference (Held-Out Verification Benchmark):**
   $$y'_{central}(t_i) = \frac{y_{i+1} - y_{i-1}}{2\Delta t}$$

---

### 2. Method B: Least Squares Linear Curve Fitting (Minimizing Squared Residuals)
Tracks underlying traffic baseline drift $\hat{y}(t) = a_0 + a_1 t$ over a sliding window of $N = 20$ historical samples by minimizing:
$$S_r = \sum_{k=1}^N \left( y_k - (a_0 + a_1 t_k) \right)^2$$

**Normal Equations Solved in Pure Python:**
$$S_t = \sum_{k=1}^N t_k, \quad S_y = \sum_{k=1}^N y_k, \quad S_{tt} = \sum_{k=1}^N t_k^2, \quad S_{ty} = \sum_{k=1}^N t_k y_k$$

$$\text{det} = N \cdot S_{tt} - (S_t)^2$$
$$a_1 = \frac{N \cdot S_{ty} - S_t \cdot S_y}{\text{det}}, \quad a_0 = \frac{S_y - a_1 \cdot S_t}{N}$$

**Window Standard Error of the Estimate ($\sigma_{est}$):**
$$\sigma_{est} = \sqrt{\frac{\sum_{k=1}^N (y_k - \hat{y}_k)^2}{\max(1, N - 2)}}$$

**Residual Anomaly Trigger:**
$$e_i = \left|y_i - \hat{y}_i\right| > \tau_r \quad (\text{default } \tau_r = 28.0\text{ pkt/s})$$

---

### 3. Anti-Corruption Window Shield (Point 8)
Standard Least Squares regressions suffer from extreme outlier leverage: an active DDoS burst would drag $a_1$ and $a_0$ drastically upward, corrupting the baseline and causing normal post-attack traffic to be falsely flagged as outages.

**Shield Mechanism:** When a sample is flagged as an anomaly, the regression training window clamps the point to the projected baseline $\hat{y}_i$, preserving baseline integrity across attacks of arbitrary duration.

---

### 4. Combined Practical Decision Rule (Point 10)
| Condition | Mathematical Criteria | Status | Automated Action |
| :--- | :--- | :--- | :--- |
| **Within Thresholds** | $\left\|y'\right\| \le \tau_d$ AND $e_i \le \tau_r$ | `NORMAL` | Forward packet telemetry |
| **Single Breach** | $\left\|y'\right\| > \tau_d$ XOR $e_i > \tau_r$ | `WARNING` | Log telemetry; tag for observation |
| **Dual Breach / Severe** | Both breached OR either $> 1.5\tau$ | `CRITICAL` | Trigger rate-limit alert & drop flood |

---

## 📋 Official 10-Point MATH 301 Breakdown

1. **Real-World Problem:** Volumetric flood and sensor outage attacks on edge IoT and network gateways.
2. **Mathematical Goal:** Pure-Python real-time anomaly detection with zero ML inference overhead.
3. **Resources:** Single Python Flask application running with minimal CPU/RAM footprint.
4. **Data Generation & Sampling:** 1-second discrete packet rates $y(t_i)$ simulating diurnal variations.
5. **Method A Math:** 1st-order, 3-point backward, and central finite differences.
6. **Method B Math:** Normal equations linear regression minimizing squared residuals.
7. **Parameter Justification:** $\tau_d = 35.0\text{ pkt/s}^2$, $\tau_r = 28.0\text{ pkt/s}$, $N = 20$ sliding window.
8. **Anti-Corruption Shield:** Window isolation preventing outlier contamination of the baseline.
9. **Verification Benchmark:** 50-sample deterministic analytical test evaluating truncation errors and confusion matrix.
10. **Combined Decision Rule:** Multi-level triage mapping numerical outputs to automated security responses.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Only dependency is Flask:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Running the Test Suite
```bash
python -m unittest test_app.py
```

---

## 📊 Interactive UI Features
- **Cyber-Lab Aesthetic:** Sleek dark-mode interface (`bg-slate-950`, glassmorphism cards, glowing status badges).
- **Dual 60fps Real-Time Charts:** Primary telemetry vs. fitted baseline $\hat{y}(t) \pm \tau_r$ and finite difference velocity $y'(t)$ vs. $\pm \tau_d$.
- **Live Anomaly Injection:** Inject DDoS Bursts (5s), Stealth Ramps (8s), Glitch Pulses (1s), or Link Outages (4s).
- **Online Confusion Matrix:** Real-time tracking of TP, FP, TN, FN, Accuracy, Precision, Recall, and F1-score.
- **CSV Log Export:** One-click instant download of timestamped telemetry data for post-lab verification.

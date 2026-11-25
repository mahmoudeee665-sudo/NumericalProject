"""
Comprehensive Unit Tests for MATH 301 - AI Network Anomaly Detector
Tests both mathematical engine correctness and Flask API reliability.
"""

import unittest
import math
from numerical_engine import (
    FiniteDifferenceEngine,
    LeastSquaresEngine,
    AnomalyEvaluator,
    ConfusionMatrixTracker,
    NetworkTrafficSimulator,
    run_50_sample_benchmark
)
from app import app, state


class TestNumericalEngine(unittest.TestCase):

    def test_finite_difference_1st_order(self):
        # f(t) = 3t + 5, f'(t) = 3
        # y(1) = 8, y(2) = 11, dt = 1
        d = FiniteDifferenceEngine.backward_first_order(11.0, 8.0, dt=1.0)
        self.assertAlmostEqual(d, 3.0, places=5)

    def test_finite_difference_3pt_backward(self):
        # f(t) = t^2, f'(t) = 2t
        # at t=3, f'(3) = 6
        # t=1: 1, t=2: 4, t=3: 9, dt = 1
        # 3pt formula: (3*9 - 4*4 + 1) / 2 = (27 - 16 + 1)/2 = 12/2 = 6.0 (Exact for quadratic!)
        d = FiniteDifferenceEngine.backward_three_point(9.0, 4.0, 1.0, dt=1.0)
        self.assertAlmostEqual(d, 6.0, places=5)

    def test_finite_difference_central(self):
        # f(t) = t^2, at t=2: f'(2) = 4
        # t=1: 1, t=3: 9, dt = 1
        # central formula: (9 - 1) / 2 = 8 / 2 = 4.0 (Exact for quadratic!)
        d = FiniteDifferenceEngine.central_difference(9.0, 1.0, dt=1.0)
        self.assertAlmostEqual(d, 4.0, places=5)

    def test_least_squares_linear_fit(self):
        # y = 2t + 10
        t_win = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_win = [12.0, 14.0, 16.0, 18.0, 20.0]
        a_0, a_1, sigma_est = LeastSquaresEngine.fit_linear(t_win, y_win)
        self.assertAlmostEqual(a_0, 10.0, places=5)
        self.assertAlmostEqual(a_1, 2.0, places=5)
        self.assertAlmostEqual(sigma_est, 0.0, places=5)

    def test_least_squares_zero_division_guard(self):
        # All t identical (det = 0)
        t_win = [2.0, 2.0, 2.0]
        y_win = [5.0, 5.0, 5.0]
        a_0, a_1, sigma_est = LeastSquaresEngine.fit_linear(t_win, y_win)
        self.assertAlmostEqual(a_0, 5.0, places=5)
        self.assertAlmostEqual(a_1, 0.0, places=5)

    def test_anomaly_evaluator_decision_matrix(self):
        # Case 1: Normal
        res_norm = AnomalyEvaluator.evaluate(
            y_curr=60.0, y_hat=60.5, y_prime_1st=2.0, tau_d=35.0, tau_r=28.0
        )
        self.assertEqual(res_norm["status"], "NORMAL")
        self.assertFalse(res_norm["detected_anomaly"])

        # Case 2: Warning (only derivative breached)
        res_warn = AnomalyEvaluator.evaluate(
            y_curr=70.0, y_hat=68.0, y_prime_1st=36.0, tau_d=35.0, tau_r=28.0
        )
        self.assertEqual(res_warn["status"], "WARNING")
        self.assertTrue(res_warn["detected_anomaly"])

        # Case 3: Critical (both breached)
        res_crit = AnomalyEvaluator.evaluate(
            y_curr=180.0, y_hat=60.0, y_prime_1st=120.0, tau_d=35.0, tau_r=28.0
        )
        self.assertEqual(res_crit["status"], "CRITICAL")
        self.assertTrue(res_crit["detected_anomaly"])

    def test_confusion_matrix_tracker(self):
        cm = ConfusionMatrixTracker()
        cm.update(detected=True, ground_truth=True)   # TP
        cm.update(detected=True, ground_truth=False)  # FP
        cm.update(detected=False, ground_truth=False) # TN
        cm.update(detected=False, ground_truth=True)  # FN
        metrics = cm.get_metrics()
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 50.0)
        self.assertAlmostEqual(metrics["precision"], 50.0)
        self.assertAlmostEqual(metrics["recall"], 50.0)

    def test_50_sample_benchmark(self):
        res = run_50_sample_benchmark(tau_d=35.0, tau_r=28.0, window_n=20)
        self.assertEqual(len(res["records"]), 50)
        self.assertGreater(res["metrics"]["accuracy"], 85.0)
        self.assertIn("mae_1st_order_backward", res["finite_diff_analysis"])


class TestFlaskEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        state.reset()

    def test_index_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"AI39 | AI Network Anomaly Detector", resp.data)

    def test_tick_lifecycle(self):
        # Cold start t=1
        r1 = self.client.get("/api/tick")
        self.assertEqual(r1.status_code, 200)
        data1 = r1.get_json()
        self.assertEqual(data1["current"]["t"], 1)

        # Sequential ticks up to t=5
        for _ in range(4):
            r = self.client.get("/api/tick")
            self.assertEqual(r.status_code, 200)
        data5 = r.get_json()
        self.assertEqual(data5["current"]["t"], 5)
        self.assertIn("sigma_est", data5["current"])

    def test_injection_route(self):
        resp = self.client.post("/api/inject", json={"mode": "ddos"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["mode"], "ddos")

        # Next tick should produce attack rate
        r_tick = self.client.get("/api/tick")
        data = r_tick.get_json()
        self.assertTrue(data["current"]["ground_truth_anomaly"])
        self.assertGreater(data["current"]["y"], 150.0)

    def test_params_update(self):
        resp = self.client.post("/api/params", json={
            "tau_d": 45.0,
            "tau_r": 32.0,
            "window_n": 25,
            "base_noise": 2.0
        })
        self.assertEqual(resp.status_code, 200)
        p = resp.get_json()["params"]
        self.assertEqual(p["tau_d"], 45.0)
        self.assertEqual(p["tau_r"], 32.0)
        self.assertEqual(p["window_n"], 25)
        self.assertEqual(p["base_noise"], 2.0)

    def test_reset_route(self):
        self.client.get("/api/tick")
        self.client.get("/api/tick")
        resp = self.client.post("/api/reset")
        self.assertEqual(resp.status_code, 200)

        # After reset, next tick should start at t=1
        r_new = self.client.get("/api/tick")
        self.assertEqual(r_new.get_json()["current"]["t"], 1)

    def test_export_csv(self):
        # Generate some ticks
        for _ in range(5):
            self.client.get("/api/tick")
        resp = self.client.get("/api/export_csv")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "text/csv")
        self.assertIn(b"Packet Rate y(t)", resp.data)


if __name__ == "__main__":
    unittest.main()

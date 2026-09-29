import sys
import json
import tempfile
import unittest
import ast
import inspect
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from p11.errors import InputContractError, PrecisionFailure, TheoremBoundaryFailure
from p11.fixtures import exact_k8_hidden_twin_fixture, exact_unknown_topology_path_fixture
from p11.approximate import fit_k8_path, fit_k9_path_records, record_noisy_k9_schedule
from p11.architecture_repair import (
    make_cyclic_network_fixture,
    make_y_tree_fixture,
    local_edge_parameters,
    permute_y_daughters,
    recover_jacobi_from_spectral_measure,
)
from p11.known_tree import default_profiles, generate_observations, recover_known_tree
from p11.known_tree import _spectral_factor_from_power
from p11.unknown_topology import (
    CountingThresholdOracle,
    recover_unknown_topology,
    threshold_from_descriptor,
)
from p11.models import make_path_system, normalized_J
from p11.scaling import J_to_L, L_to_J
import run_known_tree_morphology_validation as known_tree_validation
from p11.io import write_json
from practical_estimator.estimator import fit_all_candidate_edges, inferred_support, threshold_schur


class JsonOutputTests(unittest.TestCase):
    def test_nonfinite_values_are_serialized_as_json_null(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            write_json(path, {"finite": np.float64(2.5), "missing": float("nan"), "nested": [float("inf")]})
            parsed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, {"finite": 2.5, "missing": None, "nested": [None]})


class PracticalUnknownTopologyTests(unittest.TestCase):
    def test_no_oracle_all_edge_fit_recovers_small_unknown_support(self):
        n = 3
        edges = [(0, 1), (1, 2)]
        incidence = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
        L = np.diag([1.0, 1.2, 0.9]) + incidence @ np.diag([1.2, 0.8]) @ incidence.T
        C = np.diag([0.9, 1.1, 1.0])
        zero = np.zeros(n)
        records = []
        for growth in (0.0, 1.0):
            q = zero.copy()
            records.append({"growth_rate": growth, "q": q, "retained": tuple(range(n)),
                            "observed_threshold": threshold_schur(L, C, growth, q, None, 0)})
            for node in (1, 2):
                for load in (0.5, 2.0, 10.0):
                    q = zero.copy(); q[node] = load
                    records.append({"growth_rate": growth, "q": q, "retained": tuple(range(n)),
                                    "observed_threshold": threshold_schur(L, C, growth, q, None, 0)})
            q = zero.copy(); q[1] = q[2] = 1.0
            records.append({"growth_rate": growth, "q": q, "retained": tuple(range(n)),
                            "observed_threshold": threshold_schur(L, C, growth, q, None, 0)})
        fit = fit_all_candidate_edges(n, 0, records)
        self.assertTrue(fit.success)
        self.assertLess(np.linalg.norm(fit.L - L) / np.linalg.norm(L), 1e-4)
        self.assertLess(np.linalg.norm(fit.C - C) / np.linalg.norm(C), 1e-4)
        self.assertEqual(inferred_support(fit), set(edges))


class KnownTreeTests(unittest.TestCase):
    def test_L_to_J_to_L_round_trip(self):
        L = np.array([[3.2, -0.4, 0.0], [-0.4, 2.7, -0.6], [0.0, -0.6, 1.9]])
        C = np.diag([0.37, 1.8, 4.2])
        recovered = J_to_L(L_to_J(L, C), C)
        self.assertLess(np.linalg.norm(recovered - L) / np.linalg.norm(L), 1e-14)

    def test_random_spd_M_tree_round_trips_with_heterogeneous_C(self):
        rng = np.random.default_rng(8675309)
        for replicate in range(20):
            n = 4 + replicate % 5
            edges = [(node, int(rng.integers(0, node))) for node in range(1, n)]
            B = np.zeros((n, len(edges)))
            for edge_index, (u, v) in enumerate(edges):
                B[u, edge_index], B[v, edge_index] = -1.0, 1.0
            local = rng.uniform(0.4, 2.0, size=n)
            coupling = rng.uniform(0.2, 1.7, size=len(edges))
            L = np.diag(local) + B @ np.diag(coupling) @ B.T
            C = np.diag(np.exp(rng.uniform(-1.4, 1.4, size=n)))
            J = L_to_J(L, C)
            recovered = J_to_L(J, C)
            invsqrt = np.diag(1.0 / np.sqrt(np.diag(C)))
            old_wrong = invsqrt @ (-J) @ invsqrt
            with self.subTest(replicate=replicate):
                self.assertGreater(np.min(np.linalg.eigvalsh(L)), 0.0)
                self.assertTrue(np.all(L - np.diag(np.diag(L)) <= 1e-12))
                self.assertLess(np.linalg.norm(recovered - L) / np.linalg.norm(L), 1e-13)
                self.assertGreater(np.linalg.norm(old_wrong - L) / np.linalg.norm(L), 1e-3)

    def test_exact_generated_morphology_KnownTree_motif_zero_imperfection_recovers_L(self):
        motif = known_tree_validation.load_motif()
        self.assertIsNotNone(motif)
        J, L, C, edges, *_ = known_tree_validation.mechanics_for_motif(motif, "inverse_length")
        observations = generate_observations(
            J, edges, np.array([0.35, 0.75, 1.20, 1.75]), default_profiles(4, root=0), root=0
        )
        _, diagnostics, L_error, J_error = known_tree_validation.fit_and_score_case(
            observations, C, edges, L, J, root=0
        )
        self.assertTrue(diagnostics["optimizer_success"])
        self.assertLess(L_error, 1e-8)
        self.assertLess(J_error, 1e-8)

    def test_all_KnownTree_sensitivity_fits_route_through_canonical_transform(self):
        source = inspect.getsource(known_tree_validation)
        tree = ast.parse(source)
        callers = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        callers.setdefault(child.func.id, set()).add(node.name)
        self.assertEqual(callers.get("fit_known_tree"), {"fit_and_score_case"})
        self.assertEqual(callers.get("J_to_L"), {"relative_L_error"})
        self.assertIn("fit_and_score_case", callers)
        self.assertNotIn("invsqrtC", source)
        self.assertNotIn("invsqrt =", source)

    def test_y_tree_profiles_resolve_labelled_daughter_swap(self):
        fixture = make_y_tree_fixture()
        swapped = permute_y_daughters(fixture.J)
        L_swapped = J_to_L(swapped, fixture.C)
        swapped_local, swapped_edges = local_edge_parameters(L_swapped, fixture.edges)
        self.assertGreater(np.linalg.norm(swapped - fixture.J), 1e-3)
        self.assertTrue(np.all(swapped_local > 0))
        self.assertTrue(np.all(swapped_edges > 0))
        frequencies = np.linspace(0.35, 1.75, 4)
        obs = generate_observations(fixture.J, fixture.edges, frequencies, default_profiles(4))
        for omega in frequencies:
            G = np.linalg.inv(1j * omega * np.eye(4) - fixture.J)
            Gswap = np.linalg.inv(1j * omega * np.eye(4) - swapped)
            self.assertLess(abs(G[0, 0] - Gswap[0, 0]), 1e-13)
        profile_difference = max(
            abs(
                abs(np.linalg.inv(1j * omega * np.eye(4) - fixture.J)[0, 2]) ** 2
                - abs(np.linalg.inv(1j * omega * np.eye(4) - swapped)[0, 2]) ** 2
            )
            for omega in frequencies
        )
        self.assertGreater(profile_difference, 1e-4)
        result = recover_known_tree(obs, fixture.edges)
        self.assertEqual(obs.profiles.shape[0], 4 - 2)
        self.assertLess(np.linalg.norm(result.J - fixture.J) / np.linalg.norm(fixture.J), 1e-7)

    def test_endpoint_path_coherent_spectral_measure_control(self):
        system = make_path_system(6, 8086)
        J = normalized_J(system)
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        recovered = recover_jacobi_from_spectral_measure(eigenvalues, eigenvectors[0, :] ** 2)
        self.assertLess(np.linalg.norm(recovered - J) / np.linalg.norm(J), 1e-12)

    def test_exact_hidden_twin_rational_fixture(self):
        J, edges, observations = exact_k8_hidden_twin_fixture()
        result = recover_known_tree(observations, edges)
        self.assertEqual(len(result.reduced_denominator) - 1, 3)
        self.assertLess(np.max(np.abs(result.J - J)), 2e-7)
        self.assertLess(result.coherent_fit_error, 1e-12)

    def test_path_family_exact_recovery(self):
        for n in range(2, 8):
            with self.subTest(n=n):
                system = make_path_system(n, 200 + n)
                J = normalized_J(system)
                edges = [(i, i + 1) for i in range(n - 1)]
                w = np.linspace(0.35, 1.75, n)
                observations = generate_observations(J, edges, w, default_profiles(n))
                result = recover_known_tree(observations, edges)
                self.assertLess(np.max(np.abs(result.J - J)), 2e-5)

    def test_duplicate_frequency_is_typed_input_error(self):
        system = make_path_system(3, 21)
        edges = [(0, 1), (1, 2)]
        with self.assertRaises(InputContractError):
            generate_observations(normalized_J(system), edges, np.array([0.4, 0.4, 1.2]))

    def test_zero_noise_constrained_path_fit(self):
        system = make_path_system(4, 1212)
        edges = [(i, i + 1) for i in range(3)]
        obs = generate_observations(
            normalized_J(system), edges, np.linspace(0.4, 1.6, 4), default_profiles(4)
        )
        Lhat, diagnostics = fit_k8_path(obs, system.C)
        self.assertTrue(diagnostics["optimizer_success"])
        self.assertLess(np.linalg.norm(Lhat - system.L) / np.linalg.norm(system.L), 1e-6)

    def test_spectral_factor_rejects_nonreal_root_pairs(self):
        frequencies = np.array([1.4, 1.8, 2.2])
        row_power = frequencies**2 - 1.0
        with self.assertRaises(PrecisionFailure):
            _spectral_factor_from_power(frequencies, row_power, np.array([1.0]), 1)


class UnknownTopologyTests(unittest.TestCase):
    def test_cyclic_general_graph_joint_recovery(self):
        fixture = make_cyclic_network_fixture()
        oracle = CountingThresholdOracle(
            lambda lam, q, retained: threshold_from_descriptor(fixture.L, fixture.C, lam, q, retained)
        )
        result = recover_unknown_topology(5, oracle, growth_rate_storage=0.8)
        self.assertEqual(result.query_count, 5 * 8 // 2)
        self.assertEqual(oracle.query_count, result.query_count)
        self.assertLess(np.linalg.norm(result.L - fixture.L) / np.linalg.norm(fixture.L), 1e-7)
        self.assertLess(np.linalg.norm(result.C - fixture.C) / np.linalg.norm(fixture.C), 1e-7)
        true_edges = {tuple(sorted(edge)) for edge in fixture.edges}
        recovered_edges = {
            (i, j) for i in range(5) for j in range(i + 1, 5) if result.L[i, j] < -1e-8
        }
        self.assertEqual(recovered_edges, true_edges)

    def test_rational_fixture_has_optimal_query_count(self):
        fixture = exact_unknown_topology_path_fixture(4)
        oracle = CountingThresholdOracle(
            lambda lam, q, retained: threshold_from_descriptor(
                fixture.L, fixture.C, lam, q, retained
            )
        )
        result = recover_unknown_topology(4, oracle, growth_rate_storage=fixture.lambda_storage)
        self.assertEqual(result.query_count, 4 * 7 // 2)
        self.assertEqual(oracle.query_count, result.query_count)
        self.assertLess(np.max(np.abs(result.L - fixture.L)), 2e-7)
        self.assertLess(np.max(np.abs(result.C - fixture.C)), 2e-7)

    def test_path_scaling_family(self):
        for n in range(2, 8):
            with self.subTest(n=n):
                system = make_path_system(n, 400 + n)
                oracle = CountingThresholdOracle(
                    lambda lam, q, retained, s=system: threshold_from_descriptor(
                        s.L, s.C, lam, q, retained
                    )
                )
                result = recover_unknown_topology(n, oracle, growth_rate_storage=0.8)
                self.assertEqual(result.query_count, n * (n + 3) // 2)
                self.assertLess(np.max(np.abs(result.L - system.L)), 2e-6)
                self.assertLess(np.max(np.abs(result.C - system.C)), 2e-6)

    def test_unlinked_hidden_node_fails_explicitly(self):
        L = np.array([[1.4, -0.3, 0.0], [-0.3, 1.2, 0.0], [0.0, 0.0, 0.9]])
        C = np.eye(3)
        oracle = CountingThresholdOracle(
            lambda lam, q, retained: threshold_from_descriptor(L, C, lam, q, retained)
        )
        with self.assertRaises(TheoremBoundaryFailure):
            recover_unknown_topology(3, oracle, growth_rate_storage=1.0)

    def test_root_shunt_and_nonfinite_growth_are_rejected(self):
        system = make_path_system(3, 9191)
        q = np.zeros(3)
        q[0] = 0.1
        with self.assertRaises(InputContractError):
            threshold_from_descriptor(system.L, system.C, 0.0, q)
        with self.assertRaises(InputContractError):
            threshold_from_descriptor(system.L, system.C, np.nan, np.zeros(3))

    def test_pair_inversion_near_green_correlation_boundary(self):
        L = np.array([[2.0, -0.005, -0.005], [-0.005, 1.0, -0.99], [-0.005, -0.99, 1.0]])
        C = np.eye(3)
        oracle = CountingThresholdOracle(
            lambda lam, q, retained: threshold_from_descriptor(L, C, lam, q, retained)
        )
        from p11.unknown_topology import recover_B_from_thresholds
        result = recover_B_from_thresholds(3, oracle, 0.0)
        self.assertLess(np.linalg.norm(result.B - L) / np.linalg.norm(L), 2e-5)

    def test_zero_storage_growth_level_is_rejected(self):
        with self.assertRaises(InputContractError):
            recover_unknown_topology(3, lambda *_: 0.0, growth_rate_storage=0.0)

    def test_zero_noise_path_restricted_fit(self):
        system = make_path_system(4, 1313)
        schedule = record_noisy_k9_schedule(
            system.L,
            system.C,
            growth_rate_storage=0.8,
            rng=np.random.default_rng(3),
            noise_sd=0.0,
        )
        self.assertEqual(schedule.design_status, "complete")
        Lhat, Chat, diagnostics = fit_k9_path_records(4, schedule.records)
        self.assertTrue(diagnostics["optimizer_success"])
        self.assertLess(np.linalg.norm(Lhat - system.L) / np.linalg.norm(system.L), 1e-6)
        self.assertLess(np.linalg.norm(Chat - system.C) / np.linalg.norm(system.C), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

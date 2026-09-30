from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .k8 import K8Observations, default_profiles


@dataclass(frozen=True)
class ExactK9Fixture:
    L: np.ndarray
    C: np.ndarray
    lambda_storage: float
    sympy_L: sp.Matrix
    sympy_C: sp.Matrix


def exact_k8_hidden_twin_fixture() -> tuple[np.ndarray, list[tuple[int, int]], K8Observations]:
    """Rational K8 star fixture with a hidden non-root-cyclic mode."""
    Q = sp.Rational
    Jsp = sp.Matrix([
        [-3, Q(1, 5), Q(3, 10), Q(2, 5)],
        [Q(1, 5), -4, 0, 0],
        [Q(3, 10), 0, -4, 0],
        [Q(2, 5), 0, 0, -6],
    ])
    edges = [(0, 1), (0, 2), (0, 3)]
    frequencies_q = [Q(1, 4), Q(1, 2), Q(3, 4), Q(1, 1)]
    n = Jsp.rows
    P = default_profiles(n, root=0)
    coherent = []
    differences = np.zeros((n - 2, n), dtype=float)
    for k, w in enumerate(frequencies_q):
        G = (sp.I * w * sp.eye(n) - Jsp).inv()
        coherent.append(complex(sp.N(G[0, 0], 40)))
        row_power = [sp.simplify(G[0, v] * sp.conjugate(G[0, v])) for v in range(n)]
        for j, profile in enumerate(P):
            val = sum(sp.Rational(str(float(profile[v]))) * row_power[v] for v in range(n))
            differences[j, k] = float(sp.N(val, 40))
    observations = K8Observations(
        frequencies=np.array([float(x) for x in frequencies_q]),
        coherent_root=np.asarray(coherent, dtype=complex),
        profiles=P,
        psd_differences=differences,
    )
    J = np.asarray(Jsp.tolist(), dtype=float)
    return J, edges, observations


def exact_k9_path_fixture(n: int = 4) -> ExactK9Fixture:
    """A small rational L,C path fixture for exact-contract oracle tests."""
    if n < 2:
        raise ValueError("n must be at least 2")
    Q = sp.Rational
    local = [Q(1, 2) + Q(i, 20) for i in range(n)]
    coupling = [Q(3, 10) + Q(i, 20) for i in range(n - 1)]
    storage = [Q(4, 5) + Q(i, 10) for i in range(n)]
    L = sp.diag(*local)
    for i, edge in enumerate(coupling):
        L[i, i] += edge
        L[i + 1, i + 1] += edge
        L[i, i + 1] = L[i + 1, i] = -edge
    C = sp.diag(*storage)
    return ExactK9Fixture(
        L=np.asarray(L.tolist(), dtype=float),
        C=np.asarray(C.tolist(), dtype=float),
        lambda_storage=0.8,
        sympy_L=L,
        sympy_C=C,
    )


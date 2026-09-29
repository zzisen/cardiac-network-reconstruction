"""Canonical transforms between physical L and normalized known-tree generator J."""
from __future__ import annotations

import numpy as np


def _positive_diagonal(Cdiag: np.ndarray) -> np.ndarray:
    C = np.asarray(Cdiag, dtype=float)
    if C.ndim == 2:
        if C.shape[0] != C.shape[1] or not np.allclose(C, np.diag(np.diag(C)), atol=1e-12):
            raise ValueError("C must be a positive diagonal matrix or its diagonal vector.")
        values = np.diag(C)
    elif C.ndim == 1:
        values = C
    else:
        raise ValueError("C must be a positive diagonal matrix or its diagonal vector.")
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("C diagonal entries must be finite and strictly positive.")
    return values


def _square_matrix(matrix: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (size, size) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite {size} by {size} matrix.")
    return value


def L_to_J(L: np.ndarray, Cdiag: np.ndarray) -> np.ndarray:
    """Return ``J = -C^(-1/2) L C^(-1/2)`` for positive diagonal C."""
    c = _positive_diagonal(Cdiag)
    matrix = _square_matrix(L, len(c), "L")
    inv_sqrt = 1.0 / np.sqrt(c)
    return -(inv_sqrt[:, None] * matrix * inv_sqrt[None, :])


def J_to_L(J: np.ndarray, Cdiag: np.ndarray) -> np.ndarray:
    """Return the inverse transform ``L = C^(1/2) (-J) C^(1/2)``."""
    c = _positive_diagonal(Cdiag)
    matrix = _square_matrix(J, len(c), "J")
    sqrt = np.sqrt(c)
    return sqrt[:, None] * (-matrix) * sqrt[None, :]

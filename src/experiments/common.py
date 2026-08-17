"""Shared, non-scientific utilities for reproducible experiments."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.data_processing.data_loader import ROOT


_HASH_CHUNK_SIZE = 1024 * 1024
_FOUR_PLAYER_COUNT = 4
_FOUR_PLAYER_MASKS = 1 << _FOUR_PLAYER_COUNT
_FOUR_PLAYER_GRAND_MASK = _FOUR_PLAYER_MASKS - 1
_FOUR_PLAYER_MEMBERSHIP = np.asarray(
    [
        [
            float(mask & (1 << player) != 0)
            for player in range(_FOUR_PLAYER_COUNT)
        ]
        for mask in range(_FOUR_PLAYER_MASKS)
    ]
)


def file_signature(path: Path) -> dict[str, int | str]:
    """Return a location-independent signature for a cached input."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def signature_matches(recorded: object, path: Path) -> bool:
    """Accept current SHA-256 signatures and migrate legacy local signatures."""
    if not isinstance(recorded, dict):
        return False
    try:
        if int(recorded["size_bytes"]) != path.stat().st_size:
            return False
        if "sha256" in recorded:
            return recorded == file_signature(path)
        return int(recorded["mtime_ns"]) == path.stat().st_mtime_ns
    except (KeyError, TypeError, ValueError, OSError):
        return False


def inputs_match(
    recorded: object,
    expected: dict[str, object],
    input_paths: dict[str, Path],
) -> bool:
    """Compare a manifest input block, including legacy file signatures."""
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        return False
    for key, expected_value in expected.items():
        if key in input_paths:
            if not signature_matches(recorded[key], input_paths[key]):
                return False
        elif recorded[key] != expected_value:
            return False
    return True


def portable_path(path: Path) -> str:
    """Store repository paths relatively and external paths absolutely."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def recorded_path_matches(recorded: object, path: Path) -> bool:
    if not isinstance(recorded, str):
        return False
    candidate = Path(recorded)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        return candidate.resolve() == path.resolve()
    except OSError:
        return False


def portable_outputs(value: Any) -> Any:
    """Convert every path stored below a manifest's outputs block."""
    if isinstance(value, dict):
        return {key: portable_outputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_outputs(item) for item in value]
    if isinstance(value, str):
        return portable_path(Path(value))
    return value


def _four_player_balanced_vertices() -> np.ndarray:
    balanced_matrix = _FOUR_PLAYER_MEMBERSHIP[1:].T
    vertices: dict[tuple[float, ...], np.ndarray] = {}
    for support_size in range(1, _FOUR_PLAYER_COUNT + 1):
        for support in combinations(
            range(_FOUR_PLAYER_MASKS - 1), support_size
        ):
            matrix = balanced_matrix[:, support]
            if np.linalg.matrix_rank(matrix) != support_size:
                continue
            weights, _, _, _ = np.linalg.lstsq(
                matrix, np.ones(_FOUR_PLAYER_COUNT), rcond=None
            )
            if (
                np.max(np.abs(matrix @ weights - 1.0)) > 1e-10
                or np.min(weights) < -1e-10
            ):
                continue
            vertex = np.zeros(_FOUR_PLAYER_MASKS - 1, dtype=float)
            vertex[list(support)] = np.maximum(weights, 0.0)
            key = tuple(np.round(vertex, 12))
            vertices[key] = np.asarray(key, dtype=float)
    if not vertices:
        raise RuntimeError("no balanced extreme family found")
    return np.stack(list(vertices.values()))


_FOUR_PLAYER_BALANCED_VERTICES = _four_player_balanced_vertices()


def four_player_bondareva_gap(savings: np.ndarray) -> float:
    """Return the balancedness gap of a four-player savings game."""
    if savings.shape != (_FOUR_PLAYER_MASKS,):
        raise ValueError("expected 16 coalition values for a four-player game")
    balanced_value = float(
        np.max(_FOUR_PLAYER_BALANCED_VERTICES @ savings[1:])
    )
    return max(
        0.0,
        balanced_value - float(savings[_FOUR_PLAYER_GRAND_MASK]),
    )

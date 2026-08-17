from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.experiments.common import (
    file_signature,
    inputs_match,
    signature_matches,
)


class ExperimentCommonTests(unittest.TestCase):
    def test_signature_depends_on_content_not_modification_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"reproducible input")
            first = file_signature(path)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

            self.assertEqual(file_signature(path), first)
            self.assertTrue(signature_matches(first, path))

    def test_input_comparison_accepts_a_legacy_local_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"legacy cache")
            legacy = {
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            expected = {"input": file_signature(path), "parameter": 4}

            self.assertTrue(
                inputs_match(
                    {"input": legacy, "parameter": 4},
                    expected,
                    {"input": path},
                )
            )


if __name__ == "__main__":
    unittest.main()

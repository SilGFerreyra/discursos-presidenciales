"""One opt-in integration test against protected shared artifacts, read-only."""
from runtime import configure_runtime
configure_runtime()

import unittest
from pathlib import Path
import numpy as np
from search_logic import load_artifacts, EXPECTED_INPUT_ROWS, EMBEDDING_DIM


class ArtifactIntegrityTests(unittest.TestCase):
    def test_real_artifacts_checksums_shape_norms_and_readonly(self):
        bundle = load_artifacts(Path(__file__).resolve().parent.parent / "salida-embeddings-bge-m3")
        self.assertEqual(len(bundle.metadata), EXPECTED_INPUT_ROWS)
        self.assertEqual(bundle.embeddings.shape, (EXPECTED_INPUT_ROWS, EMBEDDING_DIM))
        self.assertEqual(bundle.embeddings.dtype, np.float32)
        self.assertIsInstance(bundle.embeddings, np.memmap)
        self.assertFalse(bundle.embeddings.flags.writeable)


if __name__ == "__main__":
    unittest.main()

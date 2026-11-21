#!/usr/bin/env python3
"""Offline tests for the Inkling 2 Mini 26B-A5B (inkling2_loop) setup scaffolding."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import setup


class Inkling2SetupTest(unittest.TestCase):
    def test_family_registered_with_32k_q8_defaults(self):
        self.assertIn("inkling2", setup.FAMILIES)
        fam = setup.FAMILIES["inkling2"]
        self.assertEqual(fam["engine"], "poly")
        self.assertEqual(fam["default_context"], 32768)
        self.assertEqual(fam["default_kv"], "q8_0")
        self.assertEqual(fam["row_bytes"], 17408)
        self.assertTrue(fam["experimental"])
        self.assertTrue(fam["windows"])
        self.assertTrue(fam["vision"])

    def test_arch_and_kv_tables(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling2"], ("inkling2_loop",))
        self.assertIn("inkling2", setup.POLY_KV["q8_0"])

    def test_primary_quants_and_pinned_hashes(self):
        fam = setup.FAMILIES["inkling2"]
        expected = ["UD-IQ4_XS", "UD-Q3_K_XL", "UD-IQ3_XXS", "UD-Q4_K_XL", "UD-Q4_K_M"]
        self.assertEqual(list(fam["models"].keys()), expected)
        for q in expected:
            d = fam["models"][q]
            local_name = setup.model_file(fam, q, 1)
            self.assertEqual(local_name, f"Inkling-2-Mini-{q}.gguf")
            self.assertEqual(d["remote_file"], f"Mainline_Support/Inkling-2-Mini-{q}.gguf_file")
            self.assertIn(local_name, fam["sha256"])
            size, sha = fam["sha256"][local_name]
            self.assertGreater(size, 11_000_000_000)
            self.assertEqual(len(sha), 64)
            # Fits comfortably in 32 GB RAM / 8 GB VRAM setup
            self.assertLessEqual(d["ram_gb"], 24)
            self.assertLessEqual(d["vram_beside_gb"], 4.0)

    def test_hf_revision_pinned(self):
        rev = setup.HF_REVISIONS.get("unsloth/Inkling-2-Mini-GGUF", "")
        self.assertEqual(len(rev), 40)
        self.assertIn(rev, setup.FAMILIES["inkling2"]["hf"])


if __name__ == "__main__":
    unittest.main()

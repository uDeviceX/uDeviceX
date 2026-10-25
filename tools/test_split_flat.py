#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "split-flat.py"

spec = importlib.util.spec_from_file_location("split_flat", SCRIPT)
split_flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_flat)


class SplitFlatTests(unittest.TestCase):
    def test_assigns_each_source_once_and_isolates_support(self):
        text = """
/* begin bundled libconfig */
libconfig implementation
/* end bundled libconfig */
/* begin bundled BOP */
bop implementation
/* end bundled BOP */
/* begin src/core/a.cpp */
core a
/* end src/core/a.cpp */
/* begin src/io/b.cpp */
io b
/* end src/io/b.cpp */
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "udx.flat.cu"
            path.write_text(text)
            units = split_flat.split_text(text)

        self.assertEqual(set(units), {"api", "support", "core", "io"})
        self.assertEqual(units["support"].count("libconfig implementation"), 1)
        self.assertEqual(units["support"].count("bop implementation"), 1)
        self.assertEqual(units["core"].count("core a"), 1)
        self.assertEqual(units["io"].count("io b"), 1)


if __name__ == "__main__":
    unittest.main()

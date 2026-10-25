#!/usr/bin/env python3
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "udx-flat"


class UdxFlatPlanTests(unittest.TestCase):
    def make_tree(self):
        root = pathlib.Path(tempfile.mkdtemp())
        for rel, text in {
            "src/main.cpp": "int main() { return 0; }\n",
            "src/core/a.cpp": "void a() {}\n",
            "src/core/b.cu": "void b() {}\n",
            "src/test/ignored.cpp": "int main() { return 1; }\n",
            "src/u/client.cpp": "int main() { return 2; }\n",
        }.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return root

    def test_plan_lists_implementation_sources_and_excludes_clients(self):
        root = self.make_tree()
        result = subprocess.run(
            [str(TOOL), "plan", "--root", str(root)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["src/core/a.cpp", "src/core/b.cu", "src/main.cpp"],
        )

    def test_plan_rejects_another_main_outside_excluded_directories(self):
        root = self.make_tree()
        extra = root / "src" / "core" / "bad.cpp"
        extra.write_text("int main() { return 3; }\n")
        result = subprocess.run(
            [str(TOOL), "plan", "--root", str(root)],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("multiple main", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()

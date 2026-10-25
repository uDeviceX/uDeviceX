#!/usr/bin/env python3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "tools" / "raw_grid.py"


class RawGridTest(unittest.TestCase):
    def test_raw_xmf_layout_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import struct

            (root / "frame.density.raw").write_bytes(struct.pack("<f", 1.0))
            (root / "frame.xmf").write_text(
                """<?xml version=\"1.0\"?>
<Xdmf Version=\"2.0\"><Domain><Grid>
<Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"2 2 2\"/>
<Geometry GeometryType=\"ORIGIN_DXDYDZ\">
<DataItem Name=\"Origin\" Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>
<DataItem Name=\"Spacing\" Dimensions=\"3\" Format=\"XML\">1 1 1</DataItem>
</Geometry>
<Attribute Name=\"density\" AttributeType=\"Scalar\" Center=\"Cell\">
<DataItem Dimensions=\"1 1 1\" NumberType=\"Float\" Precision=\"4\" Format=\"Binary\">frame.density.raw</DataItem>
</Attribute>
</Grid></Domain></Xdmf>"""
            )
            result = subprocess.run(
                [sys.executable, str(UTILITY), "check", str(root / "frame.xmf")],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("channels=density", result.stdout)
            self.assertIn("dimensions=1x1x1", result.stdout)


if __name__ == "__main__":
    unittest.main()

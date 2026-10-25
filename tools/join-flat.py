#!/usr/bin/env python3
"""Assemble editable flat units into one reviewable translation-unit artifact."""

from __future__ import annotations

import argparse
import pathlib
import re


PREFIX = re.compile(
    r"^// generated from udx\.flat\.cu\n"
    r"#define DEV_CUDA\n#define CC_RELEASE\n#define KL_RELEASE\n"
    r"#include \"udx\.api\.cuh\"\n\n",
    re.M,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".")
    parser.add_argument("--output", default="build/udx.flat.cu")
    args = parser.parse_args()

    source_dir = pathlib.Path(args.input)
    api = (source_dir / "udx.api.cuh").read_text()
    support = (source_dir / "udx.support.cu").read_text()
    libconfig = (source_dir / "udx.libconfig.cu").read_text()
    units = sorted(source_dir.glob("udx.*.cu"))
    units = [p for p in units if p.name not in {"udx.support.cu", "udx.libconfig.cu"}]

    parts = [
        "// assembled from editable flat units; generated artifact\n",
        "/* begin flat/udx.libconfig.cu */\n",
        libconfig,
        "\n/* end flat/udx.libconfig.cu */\n",
        "/* begin flat/udx.support.cu */\n",
        support,
        "\n/* end flat/udx.support.cu */\n",
        "/* begin flat/udx.api.cuh */\n",
        api,
        "\n/* end flat/udx.api.cuh */\n",
    ]
    for unit in units:
        text = PREFIX.sub("", unit.read_text(), count=1)
        parts.extend([
        f"/* begin {unit.name} */\n",
            text,
            f"\n/* end {unit.name} */\n",
        ])

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate raw grid/XMF output."""

from __future__ import annotations

import argparse
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(x) for x in value.split())


def _text(element: ET.Element) -> str:
    return " ".join((element.text or "").split())


def read_xmf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    topology = root.find(".//Topology")
    if topology is None or topology.get("TopologyType") != "3DCORECTMesh":
        raise ValueError("expected a 3DCORECTMesh topology")
    topology_dims = _ints(topology.attrib["Dimensions"])
    if len(topology_dims) != 3:
        raise ValueError("expected three topology dimensions")

    geometry = root.find(".//Geometry")
    if geometry is None:
        raise ValueError("missing Geometry metadata")
    geometry_values = {}
    for item in geometry.findall("./DataItem"):
        name = item.get("Name")
        if name in ("Origin", "Spacing"):
            geometry_values[name.lower()] = _floats(_text(item))
    if len(geometry_values.get("origin", ())) != 3 or len(geometry_values.get("spacing", ())) != 3:
        raise ValueError("Origin and Spacing must each contain three values")

    channels = []
    for attribute in root.findall(".//Attribute"):
        item = attribute.find("./DataItem")
        if item is None:
            raise ValueError(f"channel {attribute.get('Name')} has no DataItem")
        dimensions = _ints(item.attrib["Dimensions"])
        if len(dimensions) == 4 and dimensions[-1:] == (1,):
            dimensions = dimensions[:-1]
        channels.append({
            "name": attribute.attrib["Name"],
            "center": attribute.attrib.get("Center"),
            "format": item.attrib.get("Format"),
            "number_type": item.attrib.get("NumberType"),
            "precision": int(item.attrib.get("Precision", "0")),
            "dimensions": dimensions,
            "source": _text(item),
        })
    if not channels:
        raise ValueError("XMF contains no channels")

    return {
        "path": path,
        "dimensions": tuple(x - 1 for x in topology_dims),
        "topology": topology_dims,
        "origin": geometry_values["origin"],
        "spacing": geometry_values["spacing"],
        "channels": channels,
    }


def _raw_path(grid: dict, channel: dict) -> Path:
    source = Path(channel["source"])
    if source.is_absolute():
        return source
    return grid["path"].parent / source


def _raw_values(path: Path):
    with path.open("rb") as stream:
        carry = b""
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            block = carry + block
            count = len(block) - (len(block) % 4)
            for value, in struct.iter_unpack("<f", block[:count]):
                yield value
            carry = block[count:]
        if carry:
            raise ValueError(f"raw file is not a multiple of four bytes: {path}")


def _validate_metadata(grid: dict) -> None:
    expected = grid["dimensions"]
    if any(x <= 0 for x in expected):
        raise ValueError("grid dimensions must be positive")
    for channel in grid["channels"]:
        if channel["center"] != "Cell":
            raise ValueError(f"channel {channel['name']} is not cell-centered")
        if channel["format"] != "Binary":
            raise ValueError(f"channel {channel['name']} is not binary data")
        if channel["number_type"] != "Float" or channel["precision"] != 4:
            raise ValueError(f"channel {channel['name']} is not float32")
        if channel["dimensions"] != expected:
            raise ValueError(
                f"channel {channel['name']} dimensions {channel['dimensions']} != {expected}"
            )


def check_raw(path: Path) -> str:
    grid = read_xmf(path)
    _validate_metadata(grid)
    count = math.prod(grid["dimensions"])
    for channel in grid["channels"]:
        raw = _raw_path(grid, channel)
        if raw.stat().st_size != count * 4:
            raise ValueError(f"wrong byte count for {raw}")
        values = _raw_values(raw)
        for index in range(count):
            value = next(values)
            if not math.isfinite(value):
                raise ValueError(f"non-finite {channel['name']} value at {index}")
        try:
            next(values)
        except StopIteration:
            continue
        raise ValueError(f"too many values in {raw}")
    names = ",".join(channel["name"] for channel in grid["channels"])
    dims = "x".join(str(x) for x in grid["dimensions"])
    return "\n".join([
        "raw-grid-check: OK",
        f"channels={names}",
        f"dimensions={dims}",
        "ordering=Z,Y,X (X-fastest)",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("xmf", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "check":
            print(check_raw(args.xmf))
    except (OSError, RuntimeError, ValueError, ET.ParseError) as error:
        print(f"grid.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Split the checked-in flat source into linkable large translation units."""

from __future__ import annotations

import argparse
import pathlib
import re


SOURCE = re.compile(r"^/\* begin (src/[^*]+\.(?:c|cpp|cu)) \*/\s*$", re.M)
HEADER = re.compile(r"^/\* begin (src/[^*]+\.h) \*/\s*$", re.M)


def _block(text: str, begin: str, end: str) -> str:
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    return text[start:finish]


def _without_wrapper(text: str) -> str:
    text = text.replace('extern "C" {\n', "", 1)
    text = text.replace('} /* extern "C" */\n', "", 1)
    return text


def _header(text: str, path: str) -> str:
    begin = f"/* begin {path} */"
    end = f"/* end {path} */"
    return _block(text, begin, end)


def _remove_headers(text: str, paths: tuple[str, ...]) -> str:
    for path in paths:
        pattern = re.compile(
            rf"/\* begin {re.escape(path)} \*/.*?/\* end {re.escape(path)} \*/\n?",
            re.S,
        )
        text = pattern.sub("", text)
    return text


def _common_header_paths(text: str) -> tuple[str, ...]:
    selected: list[tuple[int, int, str]] = []
    for match in HEADER.finditer(text):
        path = match.group(1)
        if any(start <= match.start() < end for start, end, _ in selected):
            continue
        end_marker = f"/* end {path} */"
        end = text.find(end_marker, match.end())
        if end < 0:
            continue
        selected.append((match.start(), end + len(end_marker), path))
    selected.sort()
    return tuple(path for _, _, path in selected)


def _unit_name(path: str) -> str:
    parts = path.split("/")
    if path == "src/main.cpp":
        return "main"
    if len(parts) >= 4 and parts[1] in {"distr", "exch"}:
        return "-".join(parts[1:3])
    if len(parts) >= 5 and parts[1] in {"rbc", "mesh", "wall"}:
        return "-".join(parts[1:4])
    return parts[1]


def split_text(text: str) -> dict[str, str]:
    lib_start = "/* begin bundled libconfig */"
    lib_source = "/* begin libconfig/lib/strbuf.c */"
    lib_end = "/* end bundled libconfig */"
    bop_start = "/* begin bundled BOP */"
    bop_source = "/* begin bop/src/bop_common.cpp */"
    bop_end = "/* end bundled BOP */"

    libconfig_support = ""
    bop_support = ""
    api_parts = []
    if lib_start in text:
        lib = _block(text, lib_start, lib_end)
        lib_support = _without_wrapper(lib)
        lib_support = lib_support.replace(
            lib_source,
            "#ifndef YY_BUFFER_NEW\n#define YY_BUFFER_NEW 0\n#define YY_BUFFER_NORMAL 1\n#define YY_BUFFER_EOF_PENDING 2\n#endif\n\n"
            + lib_source,
            1,
        )
        libconfig_support = lib_support
        api_parts.append(
            _without_wrapper(lib[:lib.index(lib_source)] if lib_source in lib else lib)
        )
    if bop_start in text:
        bop = _block(text, bop_start, bop_end)
        bop_support = bop
        api_parts.append(bop[:bop.index(bop_source)] if bop_source in bop else bop)

    matches = list(SOURCE.finditer(text))
    units: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = _unit_name(match.group(1))
        units.setdefault(name, []).append(text[match.start():end])

    common_paths = _common_header_paths(text)
    common = []
    for path in common_paths:
        try:
            common.append(_header(text, path))
        except ValueError:
            pass
    api_parts.extend(common)
    prefix = (
        "// generated from udx.flat.cu\n"
        "#define DEV_CUDA\n#define CC_RELEASE\n#define KL_RELEASE\n"
        "#include \"udx.api.cuh\"\n\n"
    )
    result = {"libconfig": libconfig_support, "support": bop_support}
    result.update(
        {
            name: prefix + "\n\n".join(
                _remove_headers(part, common_paths) for part in parts
            )
            for name, parts in units.items()
        }
    )
    result["api"] = "#pragma once\n" + "\n\n".join(api_parts)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="udx.flat.cu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = pathlib.Path(args.input).read_text()
    output = pathlib.Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    units = split_text(source)
    (output / "udx.api.cuh").write_text(units.pop("api"))
    for name, content in units.items():
        (output / f"udx.{name}.cu").write_text(content)
    names = sorted(units)
    (output / "units.mk").write_text(
        "UNIT_SOURCES := " + " ".join(f"$(UNIT_DIR)/udx.{name}.cu" for name in names) + "\n"
    )
    print(" ".join(sorted(units)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

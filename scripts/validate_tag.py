#!/usr/bin/env python
"""Check a git tag belongs to the package being built.

Each package's CD workflow filters on its own tag glob, but a glob is a weak
check: `galax-interop-gala-v*` and `galax-interop-galpy-v*` are distinct, yet a
naive prefix comparison accepts one for the other. A package built under another
package's tag gets that package's version, silently.

Usage: validate_tag.py <tag> <package-name>
"""

import re
import sys

# PEP 440 covers more than this; these are the forms this project releases.
VERSION = r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+|\.dev\d+)?"


def main(argv: list[str]) -> int:
    """Return 0 if `argv[1]` is a release tag for the package `argv[2]`."""
    if len(argv) != 3:
        print(f"usage: {argv[0]} <tag> <package-name>", file=sys.stderr)
        return 2

    tag, package = argv[1], argv[2]
    # `galax.interop.gala` -> `galax-interop-gala`
    prefix = package.replace(".", "-")
    # Anchored on both ends: `galax-interop-gala-v*` must not match a
    # `galax-interop-galpy-` tag.
    pattern = rf"^{re.escape(prefix)}-v{VERSION}$"

    if re.fullmatch(pattern, tag) is None:
        print(
            f"tag {tag!r} is not a release tag for {package!r} "
            f"(expected {prefix}-v<version>)",
            file=sys.stderr,
        )
        return 1

    print(f"tag {tag!r} validated for {package!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

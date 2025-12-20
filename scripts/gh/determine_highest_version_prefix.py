#!/usr/bin/env python3
"""Determine the next version tag from existing git tags.

The script scans repository tags matching v<major>.<minor>.<patch>. It finds the
highest major.minor.patch version and increments the patch number for the next tag.

Falls back to 0.1.0 when no matching tags exist.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Iterable, Tuple

TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
FALLBACK_VERSION = (0, 1, 0)


def git_tags() -> Iterable[str]:
    """Return all git tags in the repository."""
    try:
        output = subprocess.check_output(["git", "tag"], text=True)
    except subprocess.CalledProcessError:
        return []
    return output.split()


def next_version(tags: Iterable[str]) -> Tuple[int, int, int]:
    """Compute the next version by incrementing the highest patch number."""
    versions: list[Tuple[int, int, int]] = []
    for tag in tags:
        match = TAG_PATTERN.match(tag.strip())
        if not match:
            continue
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        versions.append((major, minor, patch))
    if versions:
        highest = max(versions)
        # Increment the patch version
        return (highest[0], highest[1], highest[2] + 1)
    return FALLBACK_VERSION


def write_output(version: str) -> None:
    """Write the version to GITHUB_OUTPUT if present."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"version={version}\n")


def main() -> int:
    version_tuple = next_version(git_tags())
    version = f"{version_tuple[0]}.{version_tuple[1]}.{version_tuple[2]}"
    write_output(version)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())

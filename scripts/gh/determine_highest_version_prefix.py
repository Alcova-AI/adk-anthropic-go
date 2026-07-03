#!/usr/bin/env python3
"""Determine the next version tag from existing git tags.

The repo maintains one branch per major version line: a "vN" branch (e.g.
`v1`) produces tags with that prefix, while the default branch (`main`)
currently produces the line named by DEFAULT_BRANCH_MAJOR below.

Within a line, the script finds the highest existing v<major>.<minor>.<patch>
tag and increments its patch number. Falls back to v<major>.0.0 when no tag
exists yet for that line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Iterable, Tuple

TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
MAINT_BRANCH_PATTERN = re.compile(r"^v(\d+)$")

# The major version line that the default branch (main) currently produces
# tags for. Update this by hand at the next major migration, when today's
# `main` becomes a `vN` maintenance branch and a new default branch takes
# over as the leading line.
DEFAULT_BRANCH_MAJOR = 2


def git_tags() -> Iterable[str]:
    """Return all git tags in the repository."""
    try:
        output = subprocess.check_output(["git", "tag"], text=True)
    except subprocess.CalledProcessError:
        return []
    return output.split()


def target_major(base_ref: str) -> int:
    """Determine which major-version line a merge target branch belongs to."""
    match = MAINT_BRANCH_PATTERN.match((base_ref or "").strip())
    if match:
        return int(match.group(1))
    return DEFAULT_BRANCH_MAJOR


def next_version(tags: Iterable[str], major: int) -> Tuple[int, int, int]:
    """Compute the next version within `major` by incrementing the highest patch."""
    versions: list[Tuple[int, int, int]] = []
    for tag in tags:
        match = TAG_PATTERN.match(tag.strip())
        if not match:
            continue
        tag_major = int(match.group(1))
        if tag_major != major:
            continue
        minor = int(match.group(2))
        patch = int(match.group(3))
        versions.append((tag_major, minor, patch))
    if versions:
        highest = max(versions)
        # Increment the patch version
        return (highest[0], highest[1], highest[2] + 1)
    return (major, 0, 0)


def write_output(version: str) -> None:
    """Write the version to GITHUB_OUTPUT if present."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"version={version}\n")


def main() -> int:
    base_ref = os.environ.get("BASE_REF", "")
    major = target_major(base_ref)
    version_tuple = next_version(git_tags(), major)
    version = f"{version_tuple[0]}.{version_tuple[1]}.{version_tuple[2]}"
    write_output(version)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())

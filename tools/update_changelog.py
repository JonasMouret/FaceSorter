#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_changelog.py — Promote the [Unreleased] section in CHANGELOG.md to a new version.

Usage:
  python tools/update_changelog.py --version 0.1.1
  python tools/update_changelog.py --version 0.1.1 --changelog CHANGELOG.md
  python tools/update_changelog.py --version 0.1.1 --allow-empty
  python tools/update_changelog.py --version 0.1.1 --dry-run
  python tools/update_changelog.py --version 0.1.1 --update-version-file src/facesorter/_version.py

What it does:
- Locates "## [Unreleased]" in the changelog and moves its content into a new "## [X.Y.Z] - YYYY-MM-DD" section.
- Inserts a fresh, empty [Unreleased] scaffold at the top.
- Optionally updates src/facesorter/_version.py (__version__).
- Refuses to cut a release if Unreleased is only placeholders, unless --allow-empty is provided.

Changelog format: Keep a Changelog 1.1.0 (headings like "## [Unreleased]" and "## [x.y.z] - YYYY-MM-DD").
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
import sys

UNRELEASED_HEADER_RE = re.compile(r"^##\s+\[Unreleased\]\s*$", re.IGNORECASE)
VERSION_HEADER_RE = re.compile(r"^##\s+\[(?P<ver>[^\]]+)\]\s*(?:-\s*(?P<date>\d{4}-\d{2}-\d{2}))?\s*$")

UNRELEASED_TEMPLATE = """## [Unreleased]

### ✨ Added
- _Placeholder for new features._

### 🔄 Changed
- _Placeholder for changes in existing functionality._

### 🐛 Fixed
- _Placeholder for bug fixes._

### 🛡️ Security
- _Placeholder for security-related changes._

"""

PLACEHOLDER_BULLETS = {
    "- _Placeholder for new features._",
    "- _Placeholder for changes in existing functionality._",
    "- _Placeholder for bug fixes._",
    "- _Placeholder for security-related changes._",
}


def _strip_placeholders(block: str) -> str:
    lines = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("### "):  # keep headings but they don't count as real content
            continue
        if s in PLACEHOLDER_BULLETS:
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def bump_changelog(changelog_path: Path, new_version: str, allow_empty: bool, dry_run: bool) -> str:
    text = changelog_path.read_text(encoding="utf-8")

    # Find [Unreleased] block boundaries
    lines = text.splitlines()
    start_idx = None
    end_idx = len(lines)
    for i, ln in enumerate(lines):
        if start_idx is None and UNRELEASED_HEADER_RE.match(ln):
            start_idx = i
            continue
        if start_idx is not None and i > start_idx and ln.startswith("## ["):
            end_idx = i
            break

    if start_idx is None:
        raise SystemExit("ERROR: '## [Unreleased]' section not found in CHANGELOG.md")

    unreleased_block = "\n".join(lines[start_idx + 1 : end_idx]).strip("\n")
    content_without_placeholders = _strip_placeholders(unreleased_block)

    if not content_without_placeholders and not allow_empty:
        raise SystemExit(
            "ERROR: '[Unreleased]' only contains placeholders. Use --allow-empty to cut an empty release "
            "or add real entries under Added/Changed/Fixed/Security."
        )

    today = dt.date.today().strftime("%Y-%m-%d")
    new_version_header = f"## [{new_version}] - {today}"

    # Build the new changelog text
    before = "\n".join(lines[: start_idx]).rstrip() + "\n\n"
    after = "\n".join(lines[end_idx:]).lstrip()

    new_unreleased = UNRELEASED_TEMPLATE.rstrip() + "\n\n"
    new_release = new_version_header + "\n" + (unreleased_block.strip() + "\n\n" if unreleased_block.strip() else "\n")

    new_text = before + new_unreleased + new_release + after

    if dry_run:
        return new_text

    changelog_path.write_text(new_text, encoding="utf-8")
    return new_text


def update_version_file(version_file: Path, new_version: str, dry_run: bool) -> None:
    if not version_file.exists():
        print(f"NOTE: version file not found: {version_file}", file=sys.stderr)
        return
    txt = version_file.read_text(encoding="utf-8")
    new_txt, n = re.subn(
        r'(__version__\s*=\s*")[^"]+(")',
        rf'\1{new_version}\2',
        txt,
        count=1,
    )
    if n == 0:
        print(f"WARNING: __version__ assignment not found in {version_file}", file=sys.stderr)
    if not dry_run:
        version_file.write_text(new_txt, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Promote [Unreleased] to a new version in CHANGELOG.md")
    p.add_argument("--version", required=True, help="New version, e.g. 0.1.1")
    p.add_argument("--changelog", default="CHANGELOG.md", help="Path to changelog file")
    p.add_argument("--allow-empty", action="store_true", help="Allow releasing with only placeholders")
    p.add_argument("--dry-run", action="store_true", help="Print the result to stdout without writing files")
    p.add_argument(
        "--update-version-file",
        default=None,
        help="Path to a Python file defining __version__ (e.g., src/facesorter/_version.py)",
    )
    args = p.parse_args()

    changelog_path = Path(args.changelog)
    if not changelog_path.exists():
        raise SystemExit(f"ERROR: changelog not found: {changelog_path}")

    new_text = bump_changelog(
        changelog_path=changelog_path,
        new_version=args.version,
        allow_empty=args.allow_empty,
        dry_run=args.dry_run,
    )

    if args.update_version_file:
        update_version_file(Path(args.update_version_file), args.version, args.dry_run)

    if args.dry_run:
        print(new_text)


if __name__ == "__main__":
    main()

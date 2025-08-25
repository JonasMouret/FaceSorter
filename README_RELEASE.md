# 📦 Release Workflow Guide

This project includes a helper script [`tools/update_changelog.py`](tools/update_changelog.py)  
to streamline the release process:

- Automatically promotes the `[Unreleased]` section in `CHANGELOG.md` into a new version section.
- Inserts a fresh empty `[Unreleased]` scaffold for future work.
- Optionally updates the project `__version__` in a Python file.
- Helps keep releases consistent with [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

---

## 🚀 Typical Release Flow

1. Make sure `CHANGELOG.md` has entries under `[Unreleased]`.
2. Run the release helper:

```bash
python tools/update_changelog.py \
  --version 0.1.1 \
  --update-version-file src/facesorter/_version.py
```

3. Commit the changes:

```bash
git add CHANGELOG.md src/facesorter/_version.py
git commit -m "chore(release): 0.1.1"
```

4. Tag the release:

```bash
git tag v0.1.1
```

5. Push branch + tag to trigger CI/CD:

```bash
git push origin main --tags
```

---

## 🔧 Command Options

```bash
python tools/update_changelog.py --help
```

### Arguments

* `--version X.Y.Z`
  Target version (required).

* `--changelog PATH`
  Path to the changelog file (default: `CHANGELOG.md`).

* `--update-version-file PATH`
  Path to a Python file containing `__version__` (e.g. `src/facesorter/_version.py`).

* `--dry-run`
  Show changes on stdout without writing files.

* `--allow-empty`
  Allow creating a release even if `[Unreleased]` only contains placeholders.

---

## 📝 Example

### Dry run preview

```bash
python tools/update_changelog.py --version 0.1.1 --dry-run
```

Shows how the changelog will look without writing any file.

### Empty release (no real entries)

```bash
python tools/update_changelog.py --version 0.1.1 --allow-empty
```

---

## ✅ Best Practices

* Always review and edit `CHANGELOG.md` before cutting a release.
* Keep `CHANGELOG.md` entries human-readable (not just commit messages).
* Use **semantic versioning**:

  * `MAJOR` for breaking changes,
  * `MINOR` for new features (backwards compatible),
  * `PATCH` for bug fixes.

---

## 🔄 Suggested Automation

You can create a simple **Makefile** entry for convenience:

```makefile
release:
	@if [ -z "$(v)" ]; then echo "Usage: make release v=X.Y.Z"; exit 1; fi
	python tools/update_changelog.py --version $(v) --update-version-file src/facesorter/_version.py
	git add CHANGELOG.md src/facesorter/_version.py
	git commit -m "chore(release): $(v)"
	git tag v$(v)
	git push origin main --tags
```

Usage:

```bash
make release v=0.1.1
```

---

## 📄 Example Generated CHANGELOG Section

Before:

```markdown
## [Unreleased]

### ✨ Added
- New feature A

### 🔄 Changed
- Improved behavior of B
```

After running:

```markdown
## [Unreleased]

### ✨ Added
- _Placeholder for new features._

...

## [0.1.1] - 2025-08-25
### ✨ Added
- New feature A

### 🔄 Changed
- Improved behavior of B
```

---

With this workflow, each release is **traceable, automated, and consistent** ✨

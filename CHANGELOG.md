# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.2.0] - 2026-05-18

### ✨ Added
- **Automatic clustering mode** ("Regroupement automatique" tab): point to any folder of photos and FaceSorter automatically groups people into `Sujet_001/`, `Sujet_002/` … folders — no reference photos required.
- New `core/cluster.py`: union-find clustering on cosine similarity, no external dependency.
- New `worker/cluster_worker.py`: dedicated QThread for the clustering pipeline with graceful stop support.
- **Windows packaging**: `tools/FaceSorter.spec` is now cross-platform; produces a one-dir `.exe` bundle on Windows with HEIF DLLs collected automatically.
- New GitHub Actions workflow `windows-build.yml` (manual trigger) for Windows builds.
- New GitHub Actions workflow `release.yml`: push a `v*` tag → builds macOS Intel + ARM + Windows in parallel and publishes a GitHub Release with all three ZIPs attached.
- **App icon**: `tools/icon.icns` (macOS), `tools/icon.ico` (Windows, multi-size 16→256 px), `src/facesorter/resources/icon.png` (256×256, displayed in the Qt window at runtime).

### 🔄 Changed
- `MainWindow` refactored into two tabs using `QTabWidget` — existing "Tri guidé" behaviour is fully preserved.
- `macos-build.yml` and `windows-build.yml` now trigger on `workflow_dispatch` only (no longer on every push to `main`); production builds happen via `release.yml` on tags.
- Upscale factors extended to `(1.5, 2.0, 3.0)` in automatic mode for better detection of small or distant faces.
- Updated README: logo, CI badges, download table for all three platforms, usage guide with threshold tuning tips.

---

## [0.1.0] - 2025-08-24
### ✨ Added
- First public release of **FaceSorter** 🎉
- Cross-platform graphical interface (PySide6) for Linux, Windows, and macOS.
- Create and delete people in `people/`.
- Drag & drop photos or folders directly onto a person.
- Thumbnail preview of photos per person (double-click = open).
- Full configuration options (similarity threshold, min face size, burst window, multi-face duplication).
- Continuous background processing with progress bar.
- Move or copy sorted photos into `output_photos/<Name>/`.
- Support for HEIC/HEIF formats (via `pillow-heif`).
- **Offline mode**: InsightFace `buffalo_l` models can be bundled inside `insightface_home/`.
- Command-line interface (`facesorter`) and graphical entry point (`facesorter-gui`).
- Modular project structure under `src/facesorter/` (config, core, io, worker, gui, cli).
- GitHub Actions workflow for macOS (Intel + ARM) with automated `.app` build.
- PyInstaller `.spec` file (`tools/FaceSorter.spec`) for reproducible builds.
- `.gitignore` with sensible defaults (ignores input/output/people folders, build artifacts).
- English documentation: README with Quick Start, build instructions, usage guide.
- CHANGELOG following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

### 🐛 Known Issues
- Thumbnail display is limited to **500 images per folder** to avoid UI freezes.
- First launch can be slow (model download and InsightFace initialization).
- On macOS, the application is not yet signed/notarized (right-click → *Open* required).

---

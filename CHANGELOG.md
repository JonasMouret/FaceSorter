# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### ✨ Added
- Unit test placeholders for `io/` and `core/` modules.
- GitHub Actions workflows for Windows and Linux build pipelines (planned).
- Screenshot/GIF placeholders in the README for better presentation.

### 🔄 Changed
- Improved GUI usability with icons and better layout (planned).
- Enhanced performance when handling very large photo sets.

### 🐛 Fixed
- _Placeholder for upcoming bug fixes._

### 🛡️ Security
- _Placeholder for security-related changes._

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

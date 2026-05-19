# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.3.0] - 2026-05-19

### 🐛 Fixed
- **Doublons de sujets** : relancer l'analyse sur un dossier résultat déjà peuplé créait de nouveaux `Sujet_001`, `Sujet_002`… pouvant correspondre à des personnes différentes des dossiers existants. La numérotation repart désormais automatiquement après le dernier sujet présent.
- **"Bad CPU type" sur Mac Intel** : le build macOS était arm64 uniquement ; Rosetta 2 ne suffit pas pour les bundles PyInstaller. Un build natif x86_64 est désormais produit via le runner macos-13.
- **App bloquée par macOS ("ce fichier n'est pas pris en charge")** : passage en mode one-dir PyInstaller (plus stable avec Gatekeeper) et ajout d'une signature ad-hoc (`codesign --deep --force --sign -`) dans le CI.

### ✨ Added
- **Taille minimale de groupe** (`min_cluster_size`) : les groupes contenant moins de N photos vont dans `_Divers` plutôt que dans un dossier `Sujet_XXX` dédié — évite de créer un dossier par photo isolée.
- **Checkbox "Effacer les dossiers résultat avant l'analyse"** : supprime les dossiers `Sujet_XXX`, `_SansVisage` et `_Divers` existants avant de démarrer, pour repartir d'un état propre.
- **Bouton "⟳ Ré-analyser le résultat"** : relit toutes les photos déjà triées dans les dossiers `Sujet_XXX`, les regroupe à nouveau avec le seuil courant et réorganise les dossiers. Permet de fusionner des doublons ou corriger un mauvais regroupement sans retoucher les photos sources.
- Tous les nouveaux paramètres sont sauvegardés/restaurés entre les sessions (`QSettings`).

### 📦 Build
- Release macOS produit maintenant **deux ZIPs** : `FaceSorter-macOS-x86_64.zip` (Intel) et `FaceSorter-macOS-arm64.zip` (Apple Silicon).
- Signature ad-hoc des `.app` pour réduire les blocages Gatekeeper.
- Passage en one-dir PyInstaller sur macOS (structure `.app` standard, plus compatible).

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

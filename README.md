<div align="center">
  <img src="src/facesorter/resources/icon.png" width="160" alt="FaceSorter logo"/>

  # FaceSorter

  **Sort your photos by person automatically using face recognition.**

  [![Build macOS](https://github.com/JonasMouret/FaceSorter/actions/workflows/macos-build.yml/badge.svg)](https://github.com/JonasMouret/FaceSorter/actions/workflows/macos-build.yml)
  [![Build Windows](https://github.com/JonasMouret/FaceSorter/actions/workflows/windows-build.yml/badge.svg)](https://github.com/JonasMouret/FaceSorter/actions/workflows/windows-build.yml)
</div>

---

## ✨ Features

### Guided mode — sort by known person
- Configure `people/`, `input_photos/` and `output_photos/` directories
- Create person folders and add reference photos via the UI or drag & drop
- Thumbnail preview for each person (double-click → open in viewer)
- Continuous background processing with configurable polling interval
- Adjustable similarity threshold, min face size, burst grouping window, multi-face duplication
- Move or copy sorted photos into `output_photos/<Name>/`
- Automatic gallery rebuild whenever `people/` changes

### Automatic mode — no reference photos needed
- Point to any folder of photos
- FaceSorter detects every face and **clusters people automatically**
- Creates `Sujet_001/`, `Sujet_002/` … folders — one per person found
- Photos without a face go to `_SansVisage/`
- One adjustable parameter: the grouping similarity threshold

### General
- Modern dark UI (Qt / PySide6)
- **Offline**: InsightFace `buffalo_l` models are bundled — no internet required at runtime
- CPU by default; optional GPU acceleration (CUDA)
- HEIC / HEIF support (iPhone photos)

---

## 📥 Download

Pre-built binaries are attached to each [GitHub Release](https://github.com/JonasMouret/FaceSorter/releases):

| Platform | File |
|---|---|
| macOS Intel (x86_64) | `FaceSorter-macOS-x86_64.zip` → drag `FaceSorter.app` to Applications |
| macOS Apple Silicon (arm64) | `FaceSorter-macOS-arm64.zip` → drag `FaceSorter.app` to Applications |
| Windows 10/11 (x86_64) | `FaceSorter-Windows-x86_64.zip` → extract and run `FaceSorter.exe` |

> **macOS first launch**: right-click → **Open** (the app is unsigned).  
> For public distribution, add Apple Developer signing and notarization.

---

## 🖼 Usage

### Guided mode

1. Open the **Tri guidé** tab.
2. Set the `Personnes`, `Photos à trier` and `Résultat` directories.
3. Create a folder per person and drop a few reference photos into each.
4. Click **Démarrer** — photos are sorted into `output/<Name>/` automatically.

Output structure:
```
output_photos/
├── Alice/
├── Bob/
├── _Unknown/       ← face detected but not recognized
└── _NoFace/        ← no face found
```

> **Tip — too many unknowns?** Lower the *Seuil de reconnaissance* (try 0.38–0.42) or add more diverse reference photos per person (different angles, lighting).

### Automatic mode

1. Open the **Regroupement automatique** tab.
2. Select the source folder (any folder of photos).
3. Select the output folder.
4. Click **Analyser et regrouper**.

Output structure:
```
output/
├── Sujet_001/      ← all photos of person A
├── Sujet_002/      ← all photos of person B
└── _SansVisage/    ← photos with no face
```

Adjust the *Seuil de regroupement* if needed:
- **Too low** (< 0.40) → different people may end up in the same folder.
- **Too high** (> 0.60) → same person may be split across several folders.
- Recommended range: **0.45 – 0.55**.

---

## 📦 Requirements (development)

Python ≥ 3.10

Main dependencies (declared in [`pyproject.toml`](pyproject.toml)):

| Package | Purpose |
|---|---|
| `PySide6` | GUI |
| `insightface` | Face detection & embedding |
| `onnxruntime` | Model inference (CPU) |
| `opencv-python` | Image loading |
| `pillow` + `pillow-heif` | Image loading incl. HEIC |
| `numpy` | Embedding arithmetic |

Native libraries:

- **Linux**: `sudo apt install libheif1 libheif-dev`
- **macOS**: `brew install libheif`
- **Windows**: included automatically in the `pillow-heif` wheel

---

## 🚀 Development Installation

```bash
git clone https://github.com/JonasMouret/FaceSorter.git
cd FaceSorter
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Run the GUI:

```bash
facesorter-gui
```

Run the CLI:

```bash
facesorter \
  --people ./people \
  --input  ./input_photos \
  --output ./output_photos \
  --move \
  --ctx -1      # -1 = CPU, 0 = first GPU
```

---

## 🛠 Building with PyInstaller

### Locally

```bash
pip install pyinstaller
# Download InsightFace models first
mkdir -p insightface_home/models
curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip -d insightface_home/models/

pyinstaller --noconfirm --clean tools/FaceSorter.spec
```

Output:

| Platform | Result |
|---|---|
| macOS | `dist/FaceSorter.app` |
| Windows | `dist/FaceSorter/FaceSorter.exe` |
| Linux | `dist/FaceSorter/FaceSorter` |

### Via GitHub Actions (CI)

Three workflows are available:

| Workflow | Trigger | Produces |
|---|---|---|
| `macos-build.yml` | push to `main` | macOS Intel + ARM ZIPs |
| `windows-build.yml` | push to `main` | Windows ZIP |
| `release.yml` | push a `v*` tag | GitHub Release with all 3 ZIPs |

**To publish a release:**

```bash
git tag v1.2.0
git push origin v1.2.0
```

GitHub Actions will build all three platforms and attach the ZIPs to the release automatically.

---

## 🔒 Offline Mode

InsightFace models (`buffalo_l`) are bundled inside `insightface_home/models/buffalo_l` at build time.  
At runtime the app sets `INSIGHTFACE_HOME` to this bundled folder — no internet connection required.

---

## 📄 License

[MIT License](LICENSE) © 2025 Jonas Mouret

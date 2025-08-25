# FaceSorter

**FaceSorter** is a cross-platform application (**macOS, Windows, Linux**) to **automatically sort and move photos by person** using [InsightFace](https://github.com/deepinsight/insightface).  

---

## ✨ Features

- Modern graphical interface (Qt / PySide6)
- Configure `people/`, `input_photos/`, `output_photos` directories
- List, create, and delete people folders (`people/<Name>`)
- Drag & drop photos or folders directly onto a person
- Thumbnail preview of photos (double-click = open in viewer)
- Configurable options: similarity threshold, min face size, burst grouping window, multi-face duplication…
- Continuous background processing (polling every N seconds) with progress bar
- Move or copy sorted photos into `output_photos/<Name>/`
- Automatic gallery rebuild whenever `people/` changes
- **Offline mode**: InsightFace models (`buffalo_l`) are bundled with the app

---

## 📦 Requirements

Python ≥ 3.10  

Main Python dependencies (declared in [pyproject.toml](pyproject.toml)):

- `PySide6`
- `insightface`
- `onnxruntime` (or `onnxruntime-gpu` if you have an NVIDIA GPU)
- `opencv-python`
- `pillow`
- `pillow-heif`
- `numpy`

Native libraries:

- **Linux**: `libheif` (`sudo apt install libheif1 libheif-dev`)
- **macOS**: via Homebrew → `brew install libheif`

---

## 🚀 Development Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/JonasMouret/FaceSorter.git
cd FaceSorter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
````

This installs the project in editable mode along with its development dependencies.

---

## ▶️ Running the Application

### GUI

```bash
facesorter-gui
```

### CLI

```bash
facesorter \
  --people ./people \
  --input ./input_photos \
  --output ./output_photos \
  --move \
  --ctx -1      # -1 = CPU, 0 = first GPU if available
```

---

## ⚡ Quick Start (Demo)

Try FaceSorter quickly with dummy folders:

```bash
# Create demo folders
mkdir -p people/Alice people/Bob input_photos output_photos

# Put at least one reference photo of each person into their folder
cp demo_photos/alice1.jpg people/Alice/
cp demo_photos/bob1.jpg people/Bob/

# Place some mixed photos into input_photos
cp demo_photos/*.jpg input_photos/

# Run FaceSorter (CLI)
facesorter --people people --input input_photos --output output_photos --move

# After processing:
# - Matched photos are moved to output_photos/Alice or output_photos/Bob
# - Unknowns go to output_photos/_Unknown
# - Photos without faces go to output_photos/_NoFace
```

Or launch the GUI:

```bash
facesorter-gui
```

Select the `people/`, `input_photos/`, and `output_photos/` folders, then click **Start**.

---

## 🖼 Usage

1. Start the app → configure the paths for `people/`, `input_photos/`, `output_photos/`.
2. Add folders for each person in `people/` (via the UI or drag & drop).
3. Place a few **reference photos** of each person inside their folder.
4. Drop your unsorted photos into `input_photos/`.
5. Click **Start** → photos are automatically sorted into `output_photos/<Name>/`.

---

## 🔒 Offline Mode

InsightFace models (`buffalo_l`) can be bundled inside `insightface_home/models/buffalo_l`.
At runtime, the app forces `INSIGHTFACE_HOME` to this bundled folder → no downloads required.

---

## 🛠 Building with PyInstaller

### Local (Linux / Windows)

```bash
pip install pyinstaller
pyinstaller tools/FaceSorter.spec
```

Result:

* **Linux** → `dist/FaceSorter/`
* **Windows** → `dist/FaceSorter.exe`

### macOS (via GitHub Actions)

On Linux you cannot build `.app` bundles directly.
👉 Use the provided GitHub Actions workflow (`.github/workflows/macos-build.yml`) which builds **FaceSorter.app** on macOS runners and publishes ZIP artifacts.

Artifacts → `FaceSorter-macOS-x86_64.zip` / `FaceSorter-macOS-arm64.zip`.

---

## ⚠️ Notes

* First run on macOS: right-click → **Open** (app is unsigned).
* For public distribution, add an Apple Developer **signing and notarization** step.
* For very large photo sets, thumbnail display is capped (default: 500 images max).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
© 2025 Jonas Mouret
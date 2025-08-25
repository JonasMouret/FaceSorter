# tools/FaceSorter.spec
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(__file__).resolve().parent.parent  # repo root
SRC = ROOT / "src"
TOOLS = ROOT / "tools"

# Rendre le package importable dans ce .spec (pour lire __version__)
sys.path.insert(0, str(SRC))
try:
    from facesorter import __version__ as FS_VERSION
except Exception:
    FS_VERSION = "0.0.0"

# --- PySide6: datas (traductions) + plugins Qt ---
pyside_datas = collect_data_files(
    "PySide6",
    includes=[
        "Qt/translations/*",
        "Qt/plugins/platforms/*",
        "Qt/plugins/imageformats/*",
        "Qt/plugins/iconengines/*",
        "Qt/plugins/styles/*",
    ],
)

# --- onnxruntime (CPU ou silicon): libs dynamiques ---
# Le package s'appelle 'onnxruntime' (même pour onnxruntime-silicon), l'import python aussi.
onnx_bins = []
try:
    onnx_bins = collect_dynamic_libs("onnxruntime")
except Exception:
    onnx_bins = []

# --- pillow-heif : données éventuelles ---
pillow_heif_datas = []
try:
    pillow_heif_datas = collect_data_files("pillow_heif")
except Exception:
    pillow_heif_datas = []

# --- libheif via Homebrew (optionnel) ---
brew_lib_candidates = ["/usr/local/lib", "/opt/homebrew/lib"]
libheif_bins = []
for libdir in brew_lib_candidates:
    if os.path.isdir(libdir):
        for fn in os.listdir(libdir):
            if fn.startswith("libheif") and fn.endswith(".dylib"):
                libheif_bins.append((os.path.join(libdir, fn), "."))

# --- libomp.dylib (Intel) optionnel ---
libomp_bins = []
for libdir in ["/usr/local/lib", "/usr/local/opt/libomp/lib", "/opt/homebrew/opt/libomp/lib"]:
    if os.path.isdir(libdir):
        cand = os.path.join(libdir, "libomp.dylib")
        if os.path.exists(cand):
            libomp_bins.append((cand, "."))

# --- Modèles InsightFace embarqués (mode hors-ligne) ---
extra_datas = []
if (ROOT / "insightface_home").is_dir():
    # copie le dossier tel quel dans l'app
    extra_datas.append((str(ROOT / "insightface_home"), "insightface_home"))

# --- Hidden imports InsightFace/ONNX & libs scientifiques (dyn load) ---
hiddenimports = set()
for mod in ("insightface", "onnx", "onnxruntime", "skimage", "scipy"):
    try:
        hiddenimports.update(collect_submodules(mod))
    except Exception:
        pass

# --- Icône (facultatif) ---
ICON_PATH = TOOLS / "icon.icns"
ICON = str(ICON_PATH) if ICON_PATH.exists() else None

a = Analysis(
    # Pointe maintenant vers l'entrée GUI de ton package
    [str(SRC / "facesorter" / "gui" / "main_window.py")],
    pathex=[str(SRC)],        # <--- important pour importer le package facesorter
    binaries=onnx_bins + libheif_bins + libomp_bins,
    datas=pyside_datas + pillow_heif_datas + extra_datas,
    hiddenimports=list(hiddenimports),
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="FaceSorter",
    console=False,            # GUI only
    icon=ICON,
)

app = BUNDLE(
    exe,
    name="FaceSorter.app",
    icon=ICON,
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleIdentifier": "org.hdj-advisor.facesorter",
        "CFBundleName": "FaceSorter",
        "CFBundleShortVersionString": FS_VERSION,
        "LSApplicationCategoryType": "public.app-category.photography",
        "NSPhotoLibraryUsageDescription": "FaceSorter needs access to your photos to organize them by person.",
        "NSDocumentsFolderUsageDescription": "FaceSorter organizes photos in your folders.",
        # "NSAppSleepDisabled": True,  # Décommente si tu veux éviter l'App Nap
    },
    bundle_identifier="org.hdj-advisor.facesorter",
)

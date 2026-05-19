# tools/FaceSorter.spec — cross-platform (macOS + Windows)
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

IS_MACOS   = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# When PyInstaller runs a spec, __file__ is not defined -> use CWD
ROOT  = Path.cwd()
SRC   = ROOT / "src"
TOOLS = ROOT / "tools"

sys.path.insert(0, str(SRC))
try:
    from facesorter import __version__ as FS_VERSION
except Exception:
    FS_VERSION = "0.0.0"

# ---------------------------------------------------------------------------
# PySide6: translations + Qt plugins
# ---------------------------------------------------------------------------
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
pyside_plugins = collect_data_files(
    "PySide6.Qt",
    includes=[
        "plugins/platforms/*",
        "plugins/imageformats/*",
        "plugins/iconengines/*",
        "plugins/styles/*",
    ],
)

# ---------------------------------------------------------------------------
# onnxruntime: dynamic libs (CPU on Windows/Linux, silicon on Apple M*)
# ---------------------------------------------------------------------------
onnx_bins = []
try:
    onnx_bins = collect_dynamic_libs("onnxruntime")
except Exception:
    pass

# ---------------------------------------------------------------------------
# pillow-heif: data + DLLs (the wheel already includes libheif on Windows)
# ---------------------------------------------------------------------------
pillow_heif_datas = []
pillow_heif_bins  = []
try:
    pillow_heif_datas = collect_data_files("pillow_heif")
    pillow_heif_bins  = collect_dynamic_libs("pillow_heif")
except Exception:
    pass

# ---------------------------------------------------------------------------
# macOS-only: libheif + libomp via Homebrew
# ---------------------------------------------------------------------------
libheif_bins  = []
libomp_bins   = []
runtime_hooks = []

if IS_MACOS:
    brew_lib_candidates = ["/usr/local/lib", "/opt/homebrew/lib"]
    for libdir in brew_lib_candidates:
        if os.path.isdir(libdir):
            for fn in os.listdir(libdir):
                if fn.startswith("libheif") and fn.endswith(".dylib"):
                    libheif_bins.append((os.path.join(libdir, fn), "."))

    for libdir in [
        "/usr/local/lib",
        "/usr/local/opt/libomp/lib",
        "/opt/homebrew/opt/libomp/lib",
    ]:
        if os.path.isdir(libdir):
            cand = os.path.join(libdir, "libomp.dylib")
            if os.path.exists(cand):
                libomp_bins.append((cand, "."))

    hook = TOOLS / "runtime_hook_dylib_path.py"
    if hook.exists():
        runtime_hooks = [str(hook)]

# ---------------------------------------------------------------------------
# InsightFace offline models bundled at build time
# ---------------------------------------------------------------------------
extra_datas = []
if (ROOT / "insightface_home").is_dir():
    extra_datas.append((str(ROOT / "insightface_home"), "insightface_home"))

# App icon (PNG embedded in the package, loaded at runtime by Qt)
icon_png = SRC / "facesorter" / "resources" / "icon.png"
if icon_png.exists():
    extra_datas.append((str(icon_png), "facesorter/resources"))

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
hiddenimports = set()
for mod in ("insightface", "onnx", "onnxruntime", "skimage", "scipy", "cv2"):
    try:
        hiddenimports.update(collect_submodules(mod))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------
ICON_ICNS = TOOLS / "icon.icns"
ICON_ICO  = TOOLS / "icon.ico"
if IS_MACOS:
    ICON = str(ICON_ICNS) if ICON_ICNS.exists() else None
elif IS_WINDOWS:
    ICON = str(ICON_ICO) if ICON_ICO.exists() else None
else:
    ICON = None

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(SRC / "facesorter" / "gui" / "main_window.py")],
    pathex=[str(SRC)],
    binaries=onnx_bins + libheif_bins + libomp_bins + pillow_heif_bins,
    datas=pyside_datas + pyside_plugins + pillow_heif_datas + extra_datas,
    hiddenimports=list(hiddenimports),
    runtime_hooks=runtime_hooks,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ---------------------------------------------------------------------------
# macOS: one-dir bundle → .app (more stable than one-file with Gatekeeper)
# ---------------------------------------------------------------------------
if IS_MACOS:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="FaceSorter",
        console=False,
        icon=ICON,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="FaceSorter",
    )
    app = BUNDLE(
        coll,
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
        },
        bundle_identifier="org.hdj-advisor.facesorter",
    )

# ---------------------------------------------------------------------------
# Windows / Linux: one-dir bundle → dist/FaceSorter/ folder + FaceSorter.exe
# ---------------------------------------------------------------------------
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="FaceSorter",
        console=False,
        icon=ICON,
        version_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="FaceSorter",
    )

# FaceSorter.spec
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# --- PySide6: datas (traductions) + plugins Qt ---
# Les hooks PyInstaller gèrent souvent les plugins, mais on force les plus critiques.
pyside_datas = collect_data_files('PySide6', includes=[
    'Qt/translations/*',
    'Qt/plugins/platforms/*',
    'Qt/plugins/imageformats/*',
    'Qt/plugins/iconengines/*',
    'Qt/plugins/styles/*',
])

# --- onnxruntime (CPU) : libs dynamiques ---
# (Sur Mac Intel, utilise 'onnxruntime'; sur Apple Silicon, ce serait 'onnxruntime-silicon')
onnx_bins = collect_dynamic_libs('onnxruntime')

# --- pillow-heif : données/ressources éventuelles ---
pillow_heif_datas = collect_data_files('pillow_heif')

# --- libheif via Homebrew (optionnel) ---
# En général, pillow-heif embarque ce qu'il faut, mais on tente d'ajouter libheif si présent.
brew_lib_candidates = ["/usr/local/lib", "/opt/homebrew/lib"]
libheif_bins = []
for libdir in brew_lib_candidates:
    if os.path.isdir(libdir):
        for fn in os.listdir(libdir):
            if fn.startswith("libheif") and fn.endswith(".dylib"):
                libheif_bins.append((os.path.join(libdir, fn), "."))

# --- libomp.dylib (Intel) optionnel : utile si onnxruntime/NumPy réclame OpenMP ---
libomp_bins = []
for libdir in ["/usr/local/lib", "/usr/local/opt/libomp/lib", "/opt/homebrew/opt/libomp/lib"]:
    if os.path.isdir(libdir):
        cand = os.path.join(libdir, "libomp.dylib")
        if os.path.exists(cand):
            libomp_bins.append((cand, "."))

# --- Modèles InsightFace en local (mode hors-ligne) ---
# Place le dossier 'insightface_home' à la racine de ton projet si tu veux embarquer les modèles.
extra_datas = []
if os.path.isdir("insightface_home"):
    extra_datas.append(("insightface_home", "insightface_home"))

# --- Hidden imports InsightFace / ONNX ---
hiddenimports = set()
hiddenimports.update(collect_submodules('insightface'))
# Certains backends d'onnxruntime peuvent être chargés dynamiquement :
for mod in ('onnx', 'onnxruntime', 'skimage', 'scipy'):
    try:
        hiddenimports.update(collect_submodules(mod))
    except Exception:
        pass

a = Analysis(
    ['sort_photos_by_person.py'],
    pathex=[],
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
    name='FaceSorter',
    console=False,  # GUI only
    icon='icon.icns' if os.path.exists('icon.icns') else None,
)

app = BUNDLE(
    exe,
    name='FaceSorter.app',
    icon='icon.icns' if os.path.exists('icon.icns') else None,
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleIdentifier': 'org.hdj-advisor.facesorter',
        'CFBundleName': 'FaceSorter',
        'CFBundleShortVersionString': '1.0.0',
        'LSApplicationCategoryType': 'public.app-category.photography',
        'NSPhotoLibraryUsageDescription': 'FaceSorter needs access to your photos to organize them by person.',
        'NSDocumentsFolderUsageDescription': 'FaceSorter organizes photos in your folders.',
        # (Optionnel) Bloquer App Nap si tu veux un polling constant sans throttle :
        # 'NSAppSleepDisabled': True,
    },
    bundle_identifier='org.hdj-advisor.facesorter',
)

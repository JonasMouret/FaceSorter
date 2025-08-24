# FaceSorter.spec
# -*- mode: python ; coding: utf-8 -*-

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs
)

# Qt / PySide6 (trads, etc.)
pyside_datas = collect_data_files('PySide6', includes=['Qt/translations/*'])

# onnxruntime (dylibs)
onnx_bins = collect_dynamic_libs('onnxruntime')

# pillow_heif (datas Python)
pillow_heif_datas = collect_data_files('pillow_heif')

# libheif (Homebrew) : copie la .dylib dans l'app
brew_lib_candidates = ["/opt/homebrew/lib", "/usr/local/lib"]  # Apple Silicon / Intel
libheif_bins = []
for libdir in brew_lib_candidates:
    if os.path.isdir(libdir):
        for fn in os.listdir(libdir):
            if fn.startswith("libheif") and ".dylib" in fn:
                libheif_bins.append((os.path.join(libdir, fn), "."))

hiddenimports = []
hiddenimports += collect_submodules('insightface')

a = Analysis(
    ['sort_photos_by_person.py'],
    pathex=[],
    binaries=onnx_bins + libheif_bins,
    datas=pyside_datas + pillow_heif_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                 # GUI app
    icon='icon.icns' if os.path.exists('icon.icns') else None,
)

app = BUNDLE(
    exe,
    name='FaceSorter.app',
    icon='icon.icns' if os.path.exists('icon.icns') else None,
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleName': 'FaceSorter',
        'CFBundleDisplayName': 'FaceSorter',
        'CFBundleIdentifier': 'org.hdj-advisor.facesorter',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSApplicationCategoryType': 'public.app-category.photography',
        'NSPhotoLibraryUsageDescription': 'FaceSorter needs access to your photos to organize them by person.',
        'NSDocumentsFolderUsageDescription': 'FaceSorter organizes photos in your folders.',
    },
    bundle_identifier='org.hdj-advisor.facesorter',
)

# FaceSorter.spec
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# PySide6 data (traductions, etc.)
pyside_datas = collect_data_files('PySide6', includes=['Qt/translations/*'])

# onnxruntime dylibs
onnx_bins = collect_dynamic_libs('onnxruntime')

# pillow_heif data
pillow_heif_datas = collect_data_files('pillow_heif')

# libheif (Homebrew) — on l’embarque pour fonctionner hors-ligne chez l’utilisateur
brew_lib_candidates = ["/opt/homebrew/lib", "/usr/local/lib"]
libheif_bins = []
for libdir in brew_lib_candidates:
    if os.path.isdir(libdir):
        for fn in os.listdir(libdir):
            if fn.startswith("libheif") and fn.endswith(".dylib"):
                libheif_bins.append((os.path.join(libdir, fn), "."))

# Nos modèles embarqués : tout le dossier insightface_home
extra_datas = []
if os.path.isdir("insightface_home"):
    extra_datas.append(("insightface_home", "insightface_home"))

hiddenimports = collect_submodules('insightface')

a = Analysis(
    ['sort_photos_by_person.py'],
    pathex=[],
    binaries=onnx_bins + libheif_bins,
    datas=pyside_datas + pillow_heif_datas + extra_datas,
    hiddenimports=hiddenimports,
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
    console=False,
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
    },
    bundle_identifier='org.hdj-advisor.facesorter',
)

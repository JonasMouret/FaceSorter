# -*- mode: python ; coding: utf-8 -*-
import os, glob
from PyInstaller.utils.hooks import collect_all

# --- Collect pillow_heif (Python side)
datas, binaries, hiddenimports = [], [], []
tmp = collect_all('pillow_heif')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# --- Embarquer les .dylib libheif et co. (Intel: /usr/local/lib ; AppleSilicon: /opt/homebrew/lib)
brew_lib_dirs = ['/usr/local/lib', '/opt/homebrew/lib']
lib_patterns = [
    'libheif*.dylib',   # core
    'libde265*.dylib',  # HEIF/HEVC
    'libx265*.dylib',   # HEVC encoder (souvent requis)
    'libaom*.dylib',    # AV1
    'libdav1d*.dylib',  # AV1 decoder
]
for d in brew_lib_dirs:
    for pat in lib_patterns:
        for f in glob.glob(os.path.join(d, pat)):
            binaries.append((f, 'lib'))  # => Contents/MacOS/lib

a = Analysis(
    ['app.py'],
    pathex=['src'],                 # layout "src/"
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['tools/runtime_hook_dylib_path.py'],  # <= on le crée plus bas
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceSorter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # évite des surprises sur macOS
    console=False,           # app fenêtrée
)

# 👇 Génère une .app propre
app = BUNDLE(
    exe,
    name='FaceSorter.app',
    icon=None,               # ou 'tools/app.icns' si tu as une icône
    bundle_identifier='org.hdj-advisor.facesorter',
    info_plist={
        'NSHighResolutionCapable': True,
        # Si tu veux un jour lire la Photothèque Apple directement :
        # 'NSPhotoLibraryUsageDescription': 'FaceSorter needs access to your Photos library to index faces.',
    },
)

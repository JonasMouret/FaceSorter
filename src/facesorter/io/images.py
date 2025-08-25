# images.py
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import cv2
from PIL import Image

# --- HEIC support (safe import) ---
_HEIF_OK = False
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    _HEIF_OK = True
except Exception:
    # Pas bloquant : on pourra toujours lire .heic via read_image_cv (Pillow),
    # mais sans l’opener, certaines opérations Pillow peuvent échouer.
    _HEIF_OK = False

# Toujours en minuscules ici (p.suffix.lower() ensuite)
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}


def _norm_path(p: Path | str) -> Path:
    """Chemin POSIX normalisé (expand ~, resolve)."""
    return Path(p).expanduser().resolve()


def list_images(root: Path | str) -> List[Path]:
    """
    Retourne toutes les images sous `root` (récursif), avec filtre d'extensions
    insensible à la casse. Ne lève pas si le dossier n'existe pas.
    """
    root = _norm_path(root)
    if not root.exists() or not root.is_dir():
        return []
    out: List[Path] = []
    try:
        for p in root.rglob("*"):
            # rglob peut lever une PermissionError sur certains sous-dossiers
            try:
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    out.append(p)
            except PermissionError:
                continue
    except PermissionError:
        # Dossier partiellement illisible : on renvoie ce qu'on a pu collecter
        pass
    return out


def ensure_dir(p: Path | str) -> None:
    _norm_path(p).mkdir(parents=True, exist_ok=True)


def safe_dirname(name: str) -> str:
    import re
    n = re.sub(r"[^a-zA-Z0-9._-]", "", name.strip().replace(" ", "_"))
    return n or "Personne"


def read_image_cv(path: Path | str) -> np.ndarray:
    """
    Lecture d'image en BGR (OpenCV). Pour HEIC/HEIF : lecture via Pillow,
    puis conversion RGB->BGR. Pour les autres : imdecode robuste (chemins unicode).
    """
    path = _norm_path(path)
    suf = path.suffix.lower()
    if suf in {".heic", ".heif"}:
        # Pillow lit l’image (si pillow-heif chargé c’est plus fiable)
        im = Image.open(path).convert("RGB")
        return np.array(im)[:, :, ::-1].copy()
    # Chemins Windows/mac avec caractères spéciaux : np.fromfile + imdecode
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback : essaye Pillow au cas où imdecode échoue
        im = Image.open(path).convert("RGB")
        return np.array(im)[:, :, ::-1].copy()
    return img

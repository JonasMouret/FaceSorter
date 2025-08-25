from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

IMAGE_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".heic",".heif"}

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_dirname(name: str) -> str:
    import re
    n = re.sub(r"[^a-zA-Z0-9._-]", "", name.strip().replace(" ", "_"))
    return n or "Personne"

def read_image_cv(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".heic",".heif"}:
        im = Image.open(path).convert("RGB")
        arr = np.array(im)[:, :, ::-1].copy()  # RGB->BGR
        return arr
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return img

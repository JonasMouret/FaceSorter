from __future__ import annotations
import os, sys, logging
from pathlib import Path
import numpy as np
from numpy.linalg import norm

log = logging.getLogger(__name__)

def _bundled_insightface_home() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / "insightface_home"

def setup_models_path():
    try:
        home = _bundled_insightface_home()
        if home.exists():
            os.environ["INSIGHTFACE_HOME"] = str(home)
            log.info("Using bundled INSIGHTFACE_HOME=%s", home)
    except Exception:
        pass

class InsightService:
    def __init__(self, ctx_id: int, det_size=(640, 640)):
        setup_models_path()
        from insightface.app import FaceAnalysis  # import tardif
        self._app = FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, img_bgr: np.ndarray, min_face_size: int, upscale_factors=(1.5, 2.0)):
        faces = self._app.get(img_bgr, max_num=16)
        if not faces:
            h, w = img_bgr.shape[:2]
            for s in upscale_factors:
                up = cv2_resize(img_bgr, int(w*s), int(h*s))
                faces = self._app.get(up, max_num=16)
                if faces:
                    for f in faces:
                        f.bbox /= s; f.kps /= s
                    break
        return [f for f in faces if _min_side(f.bbox) >= min_face_size]

def cosine_sim(a: np.ndarray, b: np.ndarray, embed_norm: bool) -> float:
    if embed_norm:
        a = a / (norm(a) + 1e-8); b = b / (norm(b) + 1e-8)
    return float(np.dot(a, b))

# petits helpers locaux pour éviter de dépendre d’OpenCV ici
def _min_side(bbox) -> int:
    x1,y1,x2,y2 = bbox.astype(int)
    return min(x2-x1, y2-y1)

def cv2_resize(img, w, h):
    import cv2
    return cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)

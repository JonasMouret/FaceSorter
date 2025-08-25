from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.linalg import norm
from . import sorter as sorter_mod
from ..io.images import list_images, read_image_cv
from ..insight.service import InsightService, cosine_sim

@dataclass
class Gallery:
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    vectors: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def rebuild(self, people_dir: Path, service: InsightService, embed_norm: bool, min_face_size: int) -> int:
        self.centroids.clear(); self.vectors.clear()
        if not people_dir.exists():
            return 0
        for person_dir in people_dir.iterdir():
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            embs: List[np.ndarray] = []
            for img_path in list_images(person_dir):
                try:
                    img = read_image_cv(img_path)
                    faces = service.detect(img, min_face_size)
                    if not faces: continue
                    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    emb = f.normed_embedding if embed_norm else f.embedding
                    embs.append(emb.astype(np.float32))
                except Exception:
                    continue
            if embs:
                self.vectors[person] = embs
                c = np.mean(np.stack(embs, axis=0), axis=0)
                if embed_norm: c = c / (norm(c) + 1e-8)
                self.centroids[person] = c
        return len(self.vectors)

    def match(self, emb: np.ndarray, embed_norm: bool, topk: int) -> Tuple[Optional[str], float]:
        best_name, best_sim = None, -1.0
        for name, c in self.centroids.items():
            sim = cosine_sim(emb, c, embed_norm)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is None: return None, 0.0
        sims = [cosine_sim(emb, g, embed_norm) for g in self.vectors[best_name]]
        sims.sort(reverse=True)
        k = min(topk, len(sims))
        return best_name, float(np.mean(sims[:k]))

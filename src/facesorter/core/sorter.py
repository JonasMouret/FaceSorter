from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import shutil, hashlib, logging
import numpy as np

from ..config import SorterConfig
from ..io.images import list_images, ensure_dir, read_image_cv
from ..insight.service import InsightService
from .gallery import Gallery
from .grouping import group_by_time

log = logging.getLogger(__name__)

def dir_signature(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists(): return h.hexdigest()
    for p in sorted(list_images(root)):
        try:
            st = p.stat()
            h.update(str(p.relative_to(root)).encode())
            h.update(str(st.st_size).encode())
            h.update(str(st.st_mtime_ns).encode())
        except FileNotFoundError:
            continue
    return h.hexdigest()

class FaceSorterCore:
    def __init__(self, cfg: SorterConfig):
        self.cfg = cfg
        self.gallery = Gallery()
        self.service: Optional[InsightService] = None

    def load_service(self):
        self.service = InsightService(ctx_id=self.cfg.ctx_id)

    def rebuild_gallery(self) -> int:
        assert self.service is not None
        return self.gallery.rebuild(
            self.cfg.people_dir, self.service, self.cfg.embed_norm, self.cfg.min_face_size
        )

    def classify_group(self, group: List[Path]) -> Dict[Path, List[str]]:
        assert self.service is not None
        votes: List[str] = []
        per_file: Dict[Path, Tuple[List[Tuple[str, float]], bool]] = {}

        for img_path in group:
            targets: List[Tuple[str, float]] = []
            had_face = False
            try:
                img = read_image_cv(img_path)
                faces = self.service.detect(img, self.cfg.min_face_size, self.cfg.upscale_factors)
                if not faces:
                    per_file[img_path]=(targets, False); continue
                had_face = True
                faces_sorted = sorted(
                    faces, key=lambda f: ((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) * f.det_score, reverse=True
                )
                if self.cfg.duplicate_multi_faces:
                    for f in faces_sorted:
                        emb = f.normed_embedding if self.cfg.embed_norm else f.embedding
                        name, sim = self.gallery.match(emb, self.cfg.embed_norm, self.cfg.topk)
                        if name and sim >= self.cfg.match_threshold:
                            targets.append((name, sim)); votes.append(name)
                else:
                    f = faces_sorted[0]
                    emb = f.normed_embedding if self.cfg.embed_norm else f.embedding
                    name, sim = self.gallery.match(emb, self.cfg.embed_norm, self.cfg.topk)
                    if name:
                        targets.append((name, sim))
                        if sim >= self.cfg.match_threshold: votes.append(name)
                per_file[img_path] = (targets, had_face)
            except Exception as e:
                log.exception("classify_group failed for %s", img_path)
                per_file[img_path] = (targets, had_face)

        final_name = ""
        if votes:
            from collections import Counter
            final_name, _ = Counter(votes).most_common(1)[0]

        file_to_targets: Dict[Path, List[str]] = {}
        for img_path, (targets, had_face) in per_file.items():
            if self.cfg.duplicate_multi_faces:
                if not had_face:
                    file_to_targets[img_path] = [final_name] if final_name else [self.cfg.noface_dirname]
                else:
                    if targets:
                        strong = [n for (n, s) in targets if s >= self.cfg.match_threshold]
                        file_to_targets[img_path] = strong or ([final_name] if final_name else [self.cfg.unknown_dirname])
                    else:
                        file_to_targets[img_path] = [final_name] if final_name else [self.cfg.unknown_dirname]
            else:
                if final_name:
                    if (not had_face) or (not targets) or (targets[0][1] < (self.cfg.match_threshold - self.cfg.ambiguous_margin)):
                        file_to_targets[img_path] = [final_name]; continue
                if not had_face:
                    file_to_targets[img_path] = [self.cfg.noface_dirname]
                else:
                    if targets:
                        name, sim = targets[0]
                        if sim >= self.cfg.match_threshold: file_to_targets[img_path] = [name]
                        elif (self.cfg.match_threshold - self.cfg.ambiguous_margin) <= sim < self.cfg.match_threshold:
                            file_to_targets[img_path] = [self.cfg.unknown_dirname]
                        else:
                            file_to_targets[img_path] = [self.cfg.unknown_dirname]
                    else:
                        file_to_targets[img_path] = [self.cfg.unknown_dirname]
        return file_to_targets

    def process_files(self, files: List[Path], log_cb=lambda s: None, progress_cb=None) -> int:
        ensure_dir(self.cfg.output_dir)
        ensure_dir(self.cfg.output_dir / self.cfg.unknown_dirname)
        ensure_dir(self.cfg.output_dir / self.cfg.noface_dirname)

        groups = group_by_time(files, self.cfg.group_window_sec)
        moved = 0; total = len(files); processed = 0
        for group in groups:
            mapping = self.classify_group(group)
            for img_path, targets in mapping.items():
                for target in targets:
                    dest_dir = self.cfg.output_dir / target
                    ensure_dir(dest_dir)
                    dest_path = _unique_path(dest_dir / img_path.name)
                    try:
                        if self.cfg.move_instead_copy:
                            shutil.move(str(img_path), str(dest_path))
                            log_cb(f"[MOVE] {img_path} -> {dest_path}")
                        else:
                            shutil.copy2(str(img_path), str(dest_path))
                            log_cb(f"[COPY] {img_path} -> {dest_path}")
                        moved += 1
                    except Exception as e:
                        log_cb(f"[ERROR] {img_path} -> {dest_path}: {e}")
                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
        return moved

def _unique_path(dest: Path) -> Path:
    if not dest.exists(): return dest
    i = 1
    stem, suf = dest.stem, dest.suffix
    while (dest.parent / f"{stem}_{i}{suf}").exists():
        i += 1
    return dest.parent / f"{stem}_{i}{suf}"

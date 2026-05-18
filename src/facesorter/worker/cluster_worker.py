from __future__ import annotations
import logging
import shutil
import threading
from pathlib import Path
from typing import Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..insight.service import InsightService
from ..io.images import list_images, ensure_dir, read_image_cv
from ..core.cluster import cluster_embeddings

log = logging.getLogger(__name__)

_NOFACE_DIRNAME = "_SansVisage"


class ClusterWorker(QThread):
    log_sig = Signal(str)
    status = Signal(str)
    progress_max = Signal(int)
    progress_val = Signal(int)
    progress_txt = Signal(str)
    finished_sig = Signal(int, int)  # n_subjects, n_noface

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        threshold: float,
        min_face_size: int,
        upscale_factors: Tuple[float, ...],
        ctx_id: int = -1,
        move: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.upscale_factors = upscale_factors
        self.ctx_id = ctx_id
        self.move = move
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.log_sig.emit(f"[ERREUR] {exc}")
            log.exception("ClusterWorker failed")
            self.status.emit("Erreur")

    def _run(self) -> None:
        self.status.emit("Chargement du modèle…")
        service = InsightService(ctx_id=self.ctx_id)

        images = list_images(self.input_dir)
        if not images:
            self.log_sig.emit("[INFO] Aucune image trouvée dans le dossier source.")
            self.status.emit("Terminé")
            self.finished_sig.emit(0, 0)
            return

        self.log_sig.emit(f"[INFO] {len(images)} image(s) trouvée(s). Extraction des visages…")
        self.progress_max.emit(len(images))

        with_face: list[tuple[Path, np.ndarray]] = []
        no_face: list[Path] = []

        for i, path in enumerate(images):
            if self._stop.is_set():
                self.log_sig.emit("[INFO] Analyse interrompue.")
                self.status.emit("Arrêté")
                return
            self.progress_val.emit(i)
            self.progress_txt.emit(f"Analyse {i + 1}/{len(images)} : {path.name}")
            try:
                img = read_image_cv(path)
                faces = service.detect(img, self.min_face_size, self.upscale_factors)
                if not faces:
                    no_face.append(path)
                    self.log_sig.emit(f"[NOFACE] {path.name}")
                else:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    emb = best.normed_embedding.astype(np.float32)
                    with_face.append((path, emb))
            except Exception as exc:
                self.log_sig.emit(f"[ERREUR] {path.name} : {exc}")
                no_face.append(path)

        self.log_sig.emit(
            f"[INFO] {len(with_face)} image(s) avec visage, {len(no_face)} sans visage."
        )
        if not with_face:
            self._copy_noface(no_face)
            self.progress_val.emit(len(images))
            self.progress_txt.emit("Terminé")
            self.status.emit("Terminé")
            self.finished_sig.emit(0, len(no_face))
            return

        self.log_sig.emit("[INFO] Regroupement en cours…")
        embeddings = [e for _, e in with_face]
        labels = cluster_embeddings(embeddings, self.threshold)
        n_subjects = max(labels) + 1

        self.log_sig.emit(f"[INFO] {n_subjects} sujet(s) identifié(s).")
        ensure_dir(self.output_dir)

        for (path, _), label in zip(with_face, labels):
            if self._stop.is_set():
                self.log_sig.emit("[INFO] Analyse interrompue.")
                self.status.emit("Arrêté")
                return
            folder_name = f"Sujet_{label + 1:03d}"
            dest_dir = self.output_dir / folder_name
            ensure_dir(dest_dir)
            dest = _unique_path(dest_dir / path.name)
            try:
                if self.move:
                    shutil.move(str(path), str(dest))
                    self.log_sig.emit(f"[MOVE] {path.name} → {folder_name}/")
                else:
                    shutil.copy2(str(path), str(dest))
                    self.log_sig.emit(f"[COPY] {path.name} → {folder_name}/")
            except Exception as exc:
                self.log_sig.emit(f"[ERREUR] {path.name} : {exc}")

        self._copy_noface(no_face)
        self.progress_val.emit(len(images))
        self.progress_txt.emit("Terminé")
        self.status.emit("Terminé")
        self.finished_sig.emit(n_subjects, len(no_face))

    def _copy_noface(self, paths: list[Path]) -> None:
        if not paths:
            return
        dest_dir = self.output_dir / _NOFACE_DIRNAME
        ensure_dir(dest_dir)
        for path in paths:
            dest = _unique_path(dest_dir / path.name)
            try:
                if self.move:
                    shutil.move(str(path), str(dest))
                else:
                    shutil.copy2(str(path), str(dest))
                self.log_sig.emit(f"[NOFACE] {path.name} → {_NOFACE_DIRNAME}/")
            except Exception as exc:
                self.log_sig.emit(f"[ERREUR] {path.name} : {exc}")


def _unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 1
    stem, suf = dest.stem, dest.suffix
    while (dest.parent / f"{stem}_{i}{suf}").exists():
        i += 1
    return dest.parent / f"{stem}_{i}{suf}"

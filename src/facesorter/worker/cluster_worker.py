from __future__ import annotations
import logging
import re
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..insight.service import InsightService
from ..io.images import list_images, ensure_dir, read_image_cv
from ..core.cluster import cluster_embeddings

log = logging.getLogger(__name__)

_NOFACE_DIRNAME = "_SansVisage"
_DIVERS_DIRNAME = "_Divers"
_SUBJECT_RE = re.compile(r"^Sujet_(\d+)$")


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
        min_cluster_size: int = 1,
        clear_before_run: bool = False,
        recluster_mode: bool = False,
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
        self.min_cluster_size = min_cluster_size
        self.clear_before_run = clear_before_run
        self.recluster_mode = recluster_mode
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _existing_subject_numbers(self) -> list[int]:
        if not self.output_dir.exists():
            return []
        nums = []
        for d in self.output_dir.iterdir():
            m = _SUBJECT_RE.match(d.name)
            if m and d.is_dir():
                nums.append(int(m.group(1)))
        return sorted(nums)

    def _clear_subject_dirs(self) -> None:
        if not self.output_dir.exists():
            return
        for d in list(self.output_dir.iterdir()):
            if not d.is_dir():
                continue
            if _SUBJECT_RE.match(d.name) or d.name in (_NOFACE_DIRNAME, _DIVERS_DIRNAME):
                shutil.rmtree(d)
                self.log_sig.emit(f"[NETTOYAGE] {d.name} supprimé")

    def _gather_subject_images(self) -> list[Path]:
        """Collect images from all Sujet_XXX folders in output_dir (for recluster)."""
        images: list[Path] = []
        if not self.output_dir.exists():
            return images
        for d in sorted(self.output_dir.iterdir()):
            if d.is_dir() and _SUBJECT_RE.match(d.name):
                images.extend(list_images(d))
        return images

    def _transfer(self, src: Path, dest_dir: Path, force_move: bool = False) -> None:
        dest = _unique_path(dest_dir / src.name)
        if self.move or force_move:
            shutil.move(str(src), str(dest))
            self.log_sig.emit(f"[MOVE] {src.name} → {dest_dir.name}/")
        else:
            shutil.copy2(str(src), str(dest))
            self.log_sig.emit(f"[COPY] {src.name} → {dest_dir.name}/")

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self.status.emit("Chargement du modèle…")
        service = InsightService(ctx_id=self.ctx_id)
        ensure_dir(self.output_dir)

        # --- Gather images ---
        if self.recluster_mode:
            images = self._gather_subject_images()
            if not images:
                self.log_sig.emit("[INFO] Aucune image dans les dossiers Sujet_XXX.")
                self.status.emit("Terminé")
                self.finished_sig.emit(0, 0)
                return
            self.log_sig.emit(f"[INFO] Ré-analyse : {len(images)} image(s) dans les sujets existants.")
            self._clear_subject_dirs()
            label_offset = 0
        else:
            if self.clear_before_run:
                self._clear_subject_dirs()
                label_offset = 0
            else:
                existing = self._existing_subject_numbers()
                label_offset = max(existing) if existing else 0
                if label_offset:
                    self.log_sig.emit(
                        f"[INFO] {len(existing)} sujet(s) existant(s) détecté(s). "
                        f"Numérotation à partir de Sujet_{label_offset + 1:03d}."
                    )

            images = list_images(self.input_dir)
            if not images:
                self.log_sig.emit("[INFO] Aucune image trouvée dans le dossier source.")
                self.status.emit("Terminé")
                self.finished_sig.emit(0, 0)
                return

        self.log_sig.emit(f"[INFO] {len(images)} image(s) à analyser. Extraction des visages…")
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
            if not self.recluster_mode:
                self._copy_noface(no_face)
            self.progress_val.emit(len(images))
            self.progress_txt.emit("Terminé")
            self.status.emit("Terminé")
            self.finished_sig.emit(0, len(no_face))
            return

        # --- Cluster ---
        self.log_sig.emit("[INFO] Regroupement en cours…")
        embeddings = [e for _, e in with_face]
        raw_labels = cluster_embeddings(embeddings, self.threshold)
        n_raw = max(raw_labels) + 1

        label_counts = Counter(raw_labels)
        small_labels = {lb for lb, cnt in label_counts.items() if cnt < self.min_cluster_size}
        valid_labels = sorted(lb for lb in range(n_raw) if lb not in small_labels)
        label_remap = {old: new + 1 + label_offset for new, old in enumerate(valid_labels)}

        n_subjects = len(valid_labels)
        if small_labels:
            self.log_sig.emit(
                f"[INFO] {n_raw} groupe(s) → {n_subjects} sujet(s) valide(s), "
                f"{len(small_labels)} groupe(s) trop petit(s) → {_DIVERS_DIRNAME}"
            )
        else:
            self.log_sig.emit(f"[INFO] {n_subjects} sujet(s) identifié(s).")

        # --- Distribute files ---
        for (path, _), label in zip(with_face, raw_labels):
            if self._stop.is_set():
                self.log_sig.emit("[INFO] Analyse interrompue.")
                self.status.emit("Arrêté")
                return

            if label in small_labels:
                dest_dir = self.output_dir / _DIVERS_DIRNAME
            else:
                dest_dir = self.output_dir / f"Sujet_{label_remap[label]:03d}"

            ensure_dir(dest_dir)
            try:
                self._transfer(path, dest_dir, force_move=self.recluster_mode)
            except Exception as exc:
                self.log_sig.emit(f"[ERREUR] {path.name} : {exc}")

        if not self.recluster_mode:
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

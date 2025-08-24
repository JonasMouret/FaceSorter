#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceSorter GUI — Interface graphique multiplateforme (macOS, Windows, Linux)
pour trier/déplacer automatiquement des photos par personne à l'aide d'InsightFace.

Fonctions clés :
- Choisir les dossiers people / input / output
- Lister/créer/supprimer des personnes (dossiers) dans people/
- Drag & drop de photos/dossiers sur une personne (copie par défaut, option déplacer)
- Aperçu des photos du dossier sélectionné (vignettes ; double-clic pour ouvrir)
- Paramétrage des seuils, fenêtrage par rafale, etc.
- Traitement continu avec barre de progression
- Reconstruction automatique de la galerie quand people/ change

Dépendances (requirements) :
    PySide6, insightface, onnxruntime, opencv-python, pillow, pillow-heif, numpy
"""
from __future__ import annotations

import os, sys
# 1) Éviter que cv2 impose son dossier de plugins Qt
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
# 2) Forcer la plateforme Qt (mets "wayland" si tu es 100% en Wayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# 3) Pointer explicitement vers les plugins PySide6 (optionnel)
try:
    import PySide6, pathlib
    pyside_dir = pathlib.Path(PySide6.__file__).parent
    qt_plugins = pyside_dir / "Qt" / "plugins"
    if qt_plugins.exists():
        os.environ["QT_PLUGIN_PATH"] = str(qt_plugins)
except Exception:
    pass

import re
import time
import hashlib
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.linalg import norm
from PIL import Image, ExifTags
from PIL.ImageQt import ImageQt  # conversion PIL -> QImage
import cv2
from pillow_heif import register_heif_opener
register_heif_opener()

# InsightFace
try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print("[ERREUR] insightface introuvable : pip install insightface")
    raise

# Qt (GUI)
from PySide6.QtCore import (Qt, QThread, Signal, QSettings, QSize, QMimeData)
from PySide6.QtGui import (
    QIcon, QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QPixmap
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox, QProgressBar,
    QListWidget, QListWidgetItem, QSplitter, QAbstractItemView, QScrollArea
)
from PySide6.QtCore import QUrl

# =========================
# Utilitaires & coeur
# =========================

def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_dirname(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name or "Personne"

def read_image_cv(path: Path) -> np.ndarray:
    # Support HEIC/HEIF via Pillow (pillow-heif)
    if path.suffix.lower() in {".heic", ".heif"}:
        im = Image.open(path)         # grâce à register_heif_opener()
        im = im.convert("RGB")
        arr = np.array(im)            # RGB
        arr = arr[:, :, ::-1].copy()  # RGB -> BGR pour OpenCV
        return arr
    # Chemin standard pour JPEG/PNG/etc.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de lire {path}")
    return img

def cosine_sim(a: np.ndarray, b: np.ndarray, embed_norm: bool) -> float:
    if embed_norm:
        a = a / (norm(a) + 1e-8)
        b = b / (norm(b) + 1e-8)
    return float(np.dot(a, b))

def get_exif_datetime(filepath: Path) -> float:
    try:
        im = Image.open(filepath)
        exif = im._getexif()
        if exif:
            exif_table = {ExifTags.TAGS.get(k): v for k, v in exif.items() if k in ExifTags.TAGS}
            dto = exif_table.get("DateTimeOriginal") or exif_table.get("DateTime")
            if dto:
                import datetime
                dto = dto.replace("/", ":")
                dt = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
                return dt.timestamp()
    except Exception:
        pass
    return filepath.stat().st_mtime

def group_by_time(files: List[Path], window_sec: int) -> List[List[Path]]:
    files_sorted = sorted(files, key=get_exif_datetime)
    groups: List[List[Path]] = []
    current: List[Path] = []
    last_ts = None
    for f in files_sorted:
        ts = get_exif_datetime(f)
        if last_ts is None or abs(ts - last_ts) <= window_sec:
            current.append(f)
        else:
            groups.append(current)
            current = [f]
        last_ts = ts
    if current:
        groups.append(current)
    return groups

def dir_signature(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists():
        return h.hexdigest()
    for p in sorted(list_images(root)):
        try:
            st = p.stat()
            rel = str(p.relative_to(root))
            h.update(rel.encode("utf-8"))
            h.update(str(st.st_size).encode("ascii"))
            h.update(str(st.st_mtime_ns).encode("ascii"))
        except FileNotFoundError:
            continue
    return h.hexdigest()

@dataclass
class SorterConfig:
    people_dir: Path
    input_dir: Path
    output_dir: Path
    unknown_dirname: str = "_Unknown"
    noface_dirname: str = "_NoFace"
    min_face_size: int = 24
    topk: int = 5
    embed_norm: bool = True
    match_threshold: float = 0.45
    ambiguous_margin: float = 0.05
    group_window_sec: int = 4
    upscale_factors: Tuple[float, float] = (1.5, 2.0)
    duplicate_multi_faces: bool = False
    ctx_id: int = -1  # -1 CPU, 0 premier GPU
    poll_seconds: int = 5
    move_instead_copy: bool = True

class FaceSorterCore:
    def __init__(self, cfg: SorterConfig):
        self.cfg = cfg
        self.app: Optional[FaceAnalysis] = None
        self.gallery: Dict[str, List[np.ndarray]] = {}
        self.centroids: Dict[str, np.ndarray] = {}

    # --- InsightFace ---
    def load_app(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.cfg.ctx_id, det_size=(640, 640))

    def detect_faces_anyscale(self, img_bgr: np.ndarray):
        assert self.app is not None
        faces = self.app.get(img_bgr, max_num=16)
        if len(faces) == 0:
            h, w = img_bgr.shape[:2]
            for s in self.cfg.upscale_factors:
                up = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
                faces = self.app.get(up, max_num=16)
                if len(faces) > 0:
                    for f in faces:
                        f.bbox /= s
                        f.kps /= s
                    break
        kept = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            if min(x2 - x1, y2 - y1) >= self.cfg.min_face_size:
                kept.append(f)
        return kept

    # --- Galerie ---
    def rebuild_gallery(self):
        self.gallery.clear()
        self.centroids.clear()
        if not self.cfg.people_dir.exists():
            return 0
        for person_dir in self.cfg.people_dir.iterdir():
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            embs = []
            for img_path in list_images(person_dir):
                try:
                    img = read_image_cv(img_path)
                    faces = self.detect_faces_anyscale(img)
                    if not faces:
                        continue
                    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    emb = f.normed_embedding if self.cfg.embed_norm else f.embedding
                    embs.append(emb.astype(np.float32))
                except Exception:
                    continue
            if embs:
                self.gallery[person] = embs
                c = np.mean(np.stack(embs, axis=0), axis=0)
                if self.cfg.embed_norm:
                    c = c / (norm(c) + 1e-8)
                self.centroids[person] = c
        return len(self.gallery)

    def match_embedding(self, emb: np.ndarray):
        best_name, best_sim = None, -1.0
        for name, c in self.centroids.items():
            sim = cosine_sim(emb, c, self.cfg.embed_norm)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is None:
            return None, 0.0
        sims = [cosine_sim(emb, g, self.cfg.embed_norm) for g in self.gallery[best_name]]
        sims.sort(reverse=True)
        top = float(np.mean(sims[:min(self.cfg.topk, len(sims))]))
        return best_name, top

    # --- Classification / Traitement ---
    def classify_group(self, group: List[Path]) -> Dict[Path, List[str]]:
        votes: List[str] = []
        per_file: Dict[Path, Tuple[List[Tuple[str, float]], bool]] = {}

        for img_path in group:
            targets: List[Tuple[str, float]] = []
            had_face = False
            try:
                img = read_image_cv(img_path)
                faces = self.detect_faces_anyscale(img)
                if not faces:
                    per_file[img_path] = (targets, False)
                    continue
                had_face = True
                faces_sorted = sorted(
                    faces,
                    key=lambda f: ((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) * f.det_score,
                    reverse=True
                )
                if self.cfg.duplicate_multi_faces:
                    for f in faces_sorted:
                        emb = f.normed_embedding if self.cfg.embed_norm else f.embedding
                        name, sim = self.match_embedding(emb)
                        if name and sim >= self.cfg.match_threshold:
                            targets.append((name, sim))
                            votes.append(name)
                else:
                    f = faces_sorted[0]
                    emb = f.normed_embedding if self.cfg.embed_norm else f.embedding
                    name, sim = self.match_embedding(emb)
                    if name:
                        targets.append((name, sim))
                        if sim >= self.cfg.match_threshold:
                            votes.append(name)
                per_file[img_path] = (targets, had_face)
            except Exception:
                per_file[img_path] = (targets, had_face)

        final_name = ""
        if votes:
            from collections import Counter
            final_name, _ = Counter(votes).most_common(1)[0]

        file_to_targets: Dict[Path, List[str]] = {}
        for img_path in group:
            targets, had_face = per_file[img_path]
            if self.cfg.duplicate_multi_faces:
                if not had_face:
                    file_to_targets[img_path] = [final_name] if final_name else [self.cfg.noface_dirname]
                else:
                    if targets:
                        strong = [n for (n, s) in targets if s >= self.cfg.match_threshold]
                        if strong:
                            file_to_targets[img_path] = strong
                        else:
                            file_to_targets[img_path] = [final_name] if final_name else [self.cfg.unknown_dirname]
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
                        if sim >= self.cfg.match_threshold:
                            file_to_targets[img_path] = [name]
                        elif (self.cfg.match_threshold - self.cfg.ambiguous_margin) <= sim < self.cfg.match_threshold:
                            file_to_targets[img_path] = [self.cfg.unknown_dirname]
                        else:
                            file_to_targets[img_path] = [self.cfg.unknown_dirname]
                    else:
                        file_to_targets[img_path] = [self.cfg.unknown_dirname]
        return file_to_targets

    def process_files(self, files: List[Path], log_cb=lambda s: None, progress_cb=None):
        ensure_dir(self.cfg.output_dir)
        ensure_dir(self.cfg.output_dir / self.cfg.unknown_dirname)
        ensure_dir(self.cfg.output_dir / self.cfg.noface_dirname)

        total = len(files)
        if total == 0:
            if progress_cb:
                progress_cb(0, 0)
            return 0

        groups = group_by_time(files, self.cfg.group_window_sec)
        moved_count = 0
        processed = 0

        for group in groups:
            mapping = self.classify_group(group)
            for img_path, targets in mapping.items():
                for target in targets:
                    dest_dir = self.cfg.output_dir / target
                    ensure_dir(dest_dir)
                    dest_path = dest_dir / img_path.name
                    if dest_path.exists():
                        stem, suf = img_path.stem, img_path.suffix
                        i = 1
                        while (dest_dir / f"{stem}_{i}{suf}").exists():
                            i += 1
                        dest_path = dest_dir / f"{stem}_{i}{suf}"
                    try:
                        if self.cfg.move_instead_copy:
                            shutil.move(str(img_path), str(dest_path))
                            log_cb(f"[MOVE] {img_path} -> {dest_path}")
                        else:
                            shutil.copy2(str(img_path), str(dest_path))
                            log_cb(f"[COPY] {img_path} -> {dest_path}")
                        moved_count += 1
                    except Exception as e:
                        log_cb(f"[ERREUR] {img_path} -> {dest_path}: {e}")

                processed += 1
                if progress_cb:
                    progress_cb(processed, total)

        return moved_count

# =========================
# Worker (thread de fond)
# =========================
class SortWorker(QThread):
    log = Signal(str)
    status = Signal(str)
    gallery_built = Signal(int)
    # Progression
    progress_set_max = Signal(int)
    progress_set_value = Signal(int)
    progress_set_text = Signal(str)

    def __init__(self, cfg: SorterConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.core = FaceSorterCore(cfg)
        self._stop = threading.Event()
        self._people_sig: Optional[str] = None

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.status.emit("Chargement du modèle InsightFace…")
            self.core.load_app()
            self.status.emit("Modèle prêt.")
        except Exception as e:
            self.log.emit(f"[ERREUR] Chargement InsightFace: {e}")
            self.status.emit("Erreur modèle — thread stoppé.")
            return

        # Initialisation galerie
        self.log.emit("[Galerie] Initialisation…")
        self.progress_set_max.emit(0)  # mode busy
        self.progress_set_text.emit("Construction de la galerie…")
        n_init = self.core.rebuild_gallery()
        self.gallery_built.emit(n_init)
        if n_init == 0:
            self.log.emit("[Galerie] Aucune référence trouvée (people/<Nom>/). "
                          "Ajoute des visages nets puis je reconstruirai automatiquement.")
        else:
            self.log.emit(f"[Galerie] OK au démarrage ({n_init} personne(s)).")

        self._people_sig = dir_signature(self.cfg.people_dir)
        self.status.emit("Boucle démarrée. Surveillance en cours…")
        # remise en veille barre
        self.progress_set_max.emit(1)
        self.progress_set_value.emit(0)
        self.progress_set_text.emit("En veille")

        while not self._stop.is_set():
            try:
                # Rebuild galerie si people/ a changé
                sig = dir_signature(self.cfg.people_dir)
                if sig != self._people_sig:
                    self.log.emit("[Galerie] Changement détecté → reconstruction…")
                    self.progress_set_max.emit(0)  # busy
                    self.progress_set_text.emit("Mise à jour de la galerie…")
                    n = self.core.rebuild_gallery()
                    self.gallery_built.emit(n)
                    if n == 0:
                        self.log.emit("[Galerie] Aucune référence valide. Ajoute des visages dans people/<Nom>/.")
                    else:
                        self.log.emit(f"[Galerie] OK ({n} personne(s)).")
                    self._people_sig = sig
                    self.progress_set_max.emit(1)
                    self.progress_set_value.emit(0)
                    self.progress_set_text.emit("En veille")

                if not self.core.gallery:
                    time.sleep(self.cfg.poll_seconds)
                    continue

                # Traiter les photos en entrée
                files_now = list_images(self.cfg.input_dir)
                self.log.emit(f"[INFO] Fichiers détectés dans input: {len(files_now)}")

                total = len(files_now)
                if total > 0:
                    self.progress_set_max.emit(total)
                    self.progress_set_value.emit(0)
                    self.progress_set_text.emit(f"Traitement de {total} fichier(s)…")
                else:
                    self.progress_set_max.emit(1)
                    self.progress_set_value.emit(0)
                    self.progress_set_text.emit("En veille")

                def _progress_cb(i, n):
                    self.progress_set_value.emit(i)

                count = self.core.process_files(files_now, self.log.emit, _progress_cb)

                if count == 0:
                    self.progress_set_text.emit("En veille")
                    time.sleep(self.cfg.poll_seconds)
                else:
                    self.progress_set_text.emit("Terminé")
                    continue
            except Exception as e:
                self.log.emit(f"[ERREUR] Boucle: {e}")
                time.sleep(self.cfg.poll_seconds)

        self.status.emit("Thread arrêté.")

# =========================
# Widget liste People avec drag & drop
# =========================
class PeopleListWidget(QListWidget):
    """Liste des personnes ; accepte le drag & drop de fichiers/dossiers sur une personne."""
    dropped_files = Signal(str, list)  # person_name, [filepaths]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        item = self.itemAt(event.position().toPoint()) if hasattr(event, "position") else self.itemAt(event.pos())
        if item is None:
            event.ignore()
            return
        person = item.data(Qt.UserRole) or item.text()
        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            p = Path(u.toLocalFile())
            if p.exists():
                paths.append(str(p))
        if paths:
            self.dropped_files.emit(person, paths)
            event.acceptProposedAction()
        else:
            event.ignore()

# =========================
# GUI
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceSorter — Tri de photos par personne")
        self.setMinimumSize(QSize(1180, 760))
        self.setWindowIcon(QIcon())

        self.settings = QSettings("hdj-advisor", "FaceSorter")

        # === Colonne gauche : panneau People + vignettes ===
        left = QWidget()
        lv = QVBoxLayout(left)
        self.people_list = PeopleListWidget()
        self.people_count_label = QLabel("0 personne")
        self.people_move_on_drop_chk = QCheckBox("Déplacer lors du dépôt (sinon copie)")

        # zone de création
        create_box = QGroupBox("Créer des dossiers people")
        ch = QHBoxLayout(create_box)
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText("Noms séparés par des virgules (ex: Jonas, Alice)")
        self.create_people_btn = QPushButton("Créer")
        ch.addWidget(self.names_edit, 1)
        ch.addWidget(self.create_people_btn)

        # actions people
        ppl_actions = QHBoxLayout()
        self.refresh_people_btn = QPushButton("Rafraîchir")
        self.add_photos_btn = QPushButton("Ajouter des photos…")
        self.delete_person_btn = QPushButton("Supprimer la personne")
        ppl_actions.addWidget(self.refresh_people_btn)
        ppl_actions.addWidget(self.add_photos_btn)
        ppl_actions.addWidget(self.delete_person_btn)

        # Vignettes
        thumbs_box = QGroupBox("Photos de la personne sélectionnée")
        tv = QVBoxLayout(thumbs_box)
        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setIconSize(QSize(128, 128))
        self.thumb_list.setSpacing(8)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setSelectionMode(QAbstractItemView.SingleSelection)
        tv.addWidget(self.thumb_list, 1)

        lv.addWidget(QLabel("People (dépose des photos ici)"))
        lv.addWidget(self.people_list, 1)
        lv.addWidget(self.people_count_label)
        lv.addWidget(self.people_move_on_drop_chk)
        lv.addWidget(create_box)
        lv.addLayout(ppl_actions)
        lv.addWidget(thumbs_box, 2)

        # === Colonne droite : config + logs ===
        right = QWidget()
        v = QVBoxLayout(right)

        # Group: Dossiers
        g_paths = QGroupBox("Dossiers")
        grid = QGridLayout(g_paths)
        self.people_edit = QLineEdit()
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.b_people = QPushButton("Parcourir…")
        self.b_input = QPushButton("Parcourir…")
        self.b_output = QPushButton("Parcourir…")
        grid.addWidget(QLabel("people/ :"), 0, 0); grid.addWidget(self.people_edit, 0, 1); grid.addWidget(self.b_people, 0, 2)
        grid.addWidget(QLabel("input_photos/ :"), 1, 0); grid.addWidget(self.input_edit, 1, 1); grid.addWidget(self.b_input, 1, 2)
        grid.addWidget(QLabel("output_photos/ :"), 2, 0); grid.addWidget(self.output_edit, 2, 1); grid.addWidget(self.b_output, 2, 2)
        v.addWidget(g_paths)

        # Group: paramètres
        g_set = QGroupBox("Paramètres")
        g = QGridLayout(g_set)
        self.threshold_spin = QDoubleSpinBox(); self.threshold_spin.setRange(0.0, 1.0); self.threshold_spin.setSingleStep(0.01)
        self.ambig_spin = QDoubleSpinBox(); self.ambig_spin.setRange(0.0, 0.5); self.ambig_spin.setSingleStep(0.01)
        self.minface_spin = QSpinBox(); self.minface_spin.setRange(8, 256)
        self.groupwin_spin = QSpinBox(); self.groupwin_spin.setRange(0, 60)
        self.poll_spin = QSpinBox(); self.poll_spin.setRange(1, 60)
        self.ctx_combo = QComboBox(); self.ctx_combo.addItems(["CPU (ctx=-1)", "GPU #0 (ctx=0)"])
        self.dup_faces_chk = QCheckBox("Dupliquer la photo pour chaque personne reconnue")
        self.move_chk = QCheckBox("Déplacer (au lieu de copier)")
        g.addWidget(QLabel("Seuil de match (cosine) :"), 0, 0); g.addWidget(self.threshold_spin, 0, 1)
        g.addWidget(QLabel("Marge ambiguë :"), 0, 2); g.addWidget(self.ambig_spin, 0, 3)
        g.addWidget(QLabel("Taille min visage (px) :"), 1, 0); g.addWidget(self.minface_spin, 1, 1)
        g.addWidget(QLabel("Fenêtre rafale (s) :"), 1, 2); g.addWidget(self.groupwin_spin, 1, 3)
        g.addWidget(QLabel("Période de poll (s) :"), 2, 0); g.addWidget(self.poll_spin, 2, 1)
        g.addWidget(QLabel("Contexte (CPU/GPU) :"), 2, 2); g.addWidget(self.ctx_combo, 2, 3)
        g.addWidget(self.dup_faces_chk, 3, 0, 1, 4)
        g.addWidget(self.move_chk, 4, 0, 1, 4)
        v.addWidget(g_set)

        # Controls
        h_ctrl = QHBoxLayout()
        self.build_gallery_btn = QPushButton("(Re)construire la galerie maintenant")
        self.open_people_btn = QPushButton("Ouvrir people/")
        self.open_input_btn = QPushButton("Ouvrir input/")
        self.open_output_btn = QPushButton("Ouvrir output/")
        self.start_btn = QPushButton("Démarrer")
        self.stop_btn = QPushButton("Arrêter"); self.stop_btn.setEnabled(False)
        h_ctrl.addWidget(self.build_gallery_btn)
        h_ctrl.addStretch(1)
        h_ctrl.addWidget(self.open_people_btn)
        h_ctrl.addWidget(self.open_input_btn)
        h_ctrl.addWidget(self.open_output_btn)
        h_ctrl.addStretch(1)
        h_ctrl.addWidget(self.start_btn)
        h_ctrl.addWidget(self.stop_btn)
        v.addLayout(h_ctrl)

        # Progress + Journal
        self.progress = QProgressBar(); self.progress.setMinimum(0); self.progress.setMaximum(1)
        self.progress.setValue(0); self.progress.setTextVisible(True); self.progress.setFormat("En veille")
        v.addWidget(self.progress)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        v.addWidget(QLabel("Journal")); v.addWidget(self.log, 1)

        # Splitter gauche/droite
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 760])
        self.setCentralWidget(splitter)

        # Signals
        self.b_people.clicked.connect(lambda: self.select_dir(self.people_edit))
        self.b_input.clicked.connect(lambda: self.select_dir(self.input_edit))
        self.b_output.clicked.connect(lambda: self.select_dir(self.output_edit))
        self.create_people_btn.clicked.connect(self.create_people_folders)
        self.refresh_people_btn.clicked.connect(self.refresh_people_list)
        self.add_photos_btn.clicked.connect(self.add_photos_to_selected)
        self.delete_person_btn.clicked.connect(self.delete_selected_person)
        self.build_gallery_btn.clicked.connect(self.build_gallery_now)
        self.open_people_btn.clicked.connect(lambda: self.open_dir(self.people_edit.text()))
        self.open_input_btn.clicked.connect(lambda: self.open_dir(self.input_edit.text()))
        self.open_output_btn.clicked.connect(lambda: self.open_dir(self.output_edit.text()))
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)

        # Drag & drop depuis la liste
        self.people_list.dropped_files.connect(self.handle_people_drop)
        self.people_list.currentItemChanged.connect(self.on_person_selection_changed)
        self.thumb_list.itemDoubleClicked.connect(self.open_thumbnail_item)

        # Defaults & settings
        self.load_settings()
        self.worker: Optional[SortWorker] = None

        # Peupler la liste au démarrage
        self.refresh_people_list()

    # === Helpers généraux ===
    def load_settings(self):
        self.people_edit.setText(self.settings.value("people_dir", str(Path.cwd()/"people")))
        self.input_edit.setText(self.settings.value("input_dir", str(Path.cwd()/"input_photos")))
        self.output_edit.setText(self.settings.value("output_dir", str(Path.cwd()/"output_photos")))
        self.threshold_spin.setValue(float(self.settings.value("match_threshold", 0.45)))
        self.ambig_spin.setValue(float(self.settings.value("ambiguous_margin", 0.05)))
        self.minface_spin.setValue(int(self.settings.value("min_face_size", 24)))
        self.groupwin_spin.setValue(int(self.settings.value("group_window_sec", 4)))
        self.poll_spin.setValue(int(self.settings.value("poll_seconds", 5)))
        ctx = int(self.settings.value("ctx_id", -1))
        self.ctx_combo.setCurrentIndex(0 if ctx == -1 else 1)
        self.dup_faces_chk.setChecked(self.settings.value("duplicate_multi_faces", "false") == "true")
        self.move_chk.setChecked(self.settings.value("move_instead_copy", "true") == "true")
        self.people_move_on_drop_chk.setChecked(self.settings.value("move_on_drop", "false") == "true")

    def save_settings(self):
        self.settings.setValue("people_dir", self.people_edit.text())
        self.settings.setValue("input_dir", self.input_edit.text())
        self.settings.setValue("output_dir", self.output_edit.text())
        self.settings.setValue("match_threshold", self.threshold_spin.value())
        self.settings.setValue("ambiguous_margin", self.ambig_spin.value())
        self.settings.setValue("min_face_size", self.minface_spin.value())
        self.settings.setValue("group_window_sec", self.groupwin_spin.value())
        self.settings.setValue("poll_seconds", self.poll_spin.value())
        ctx_id = -1 if self.ctx_combo.currentIndex() == 0 else 0
        self.settings.setValue("ctx_id", ctx_id)
        self.settings.setValue("duplicate_multi_faces", "true" if self.dup_faces_chk.isChecked() else "false")
        self.settings.setValue("move_instead_copy", "true" if self.move_chk.isChecked() else "false")
        self.settings.setValue("move_on_drop", "true" if self.people_move_on_drop_chk.isChecked() else "false")

    def select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Choisir un dossier", edit.text() or str(Path.cwd()))
        if d:
            edit.setText(d)
            self.save_settings()
            if edit is self.people_edit:
                self.refresh_people_list()

    def open_dir(self, path_str: str):
        if not path_str:
            return
        p = Path(path_str)
        ensure_dir(p)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # === Panneau People ===
    def refresh_people_list(self):
        people_dir = Path(self.people_edit.text())
        ensure_dir(people_dir)
        self.people_list.clear()
        count_people = 0
        for sub in sorted(people_dir.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name
            n_imgs = len(list_images(sub))
            item = QListWidgetItem(f"{name}  —  {n_imgs} image(s)")
            item.setData(Qt.UserRole, name)
            self.people_list.addItem(item)
            count_people += 1
        self.people_count_label.setText(f"{count_people} personne(s)")
        # rafraîchit les vignettes si la sélection actuelle existe encore
        self.on_person_selection_changed(self.people_list.currentItem(), None)

    def create_people_folders(self):
        names_line = self.names_edit.text().strip()
        if not names_line:
            QMessageBox.information(self, "Info", "Entrez des noms séparés par des virgules.")
            return
        people_dir = Path(self.people_edit.text())
        ensure_dir(people_dir)
        created = []
        for raw in names_line.split(','):
            n = raw.strip()
            if not n:
                continue
            dn = safe_dirname(n)
            ensure_dir(people_dir / dn)
            created.append(dn)
        if created:
            self.append_log(f"[people] Dossiers créés : {', '.join(created)}")
            self.refresh_people_list()
        else:
            self.append_log("[people] Aucun dossier créé.")
        self.names_edit.clear()

    def delete_selected_person(self):
        item = self.people_list.currentItem()
        if not item:
            QMessageBox.information(self, "Info", "Sélectionne une personne à supprimer.")
            return
        name = item.data(Qt.UserRole) or item.text().split("—",1)[0].strip()
        people_dir = Path(self.people_edit.text())
        target = people_dir / name
        if not target.exists() or not target.is_dir():
            QMessageBox.warning(self, "Attention", f"Le dossier {target} n'existe pas.")
            return
        # confirmation
        ret = QMessageBox.question(
            self, "Confirmer la suppression",
            f"Supprimer définitivement le dossier '{name}' et tout son contenu ?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(target)
            self.append_log(f"[people] Dossier supprimé : {target}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Suppression échouée : {e}")
            return
        # rafraîchir
        self.refresh_people_list()
        self.thumb_list.clear()

    def _gather_image_paths(self, paths: List[str]) -> List[Path]:
        imgs: List[Path] = []
        for pstr in paths:
            p = Path(pstr)
            if p.is_dir():
                imgs.extend(list_images(p))
            elif p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".heic",".heif"}:
                imgs.append(p)
        return imgs

    def handle_people_drop(self, person_name: str, paths: List[str]):
        """Drag & drop: copie/déplace des fichiers/dossiers dans people/<person_name>/"""
        people_dir = Path(self.people_edit.text())
        target_dir = people_dir / safe_dirname(person_name)
        ensure_dir(target_dir)

        imgs = self._gather_image_paths(paths)
        if not imgs:
            QMessageBox.information(self, "Info", "Aucune image reconnue dans le dépôt.")
            return

        move = self.people_move_on_drop_chk.isChecked()
        for src in imgs:
            dest = target_dir / src.name
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                i = 1
                while (target_dir / f"{stem}_{i}{suf}").exists():
                    i += 1
                dest = target_dir / f"{stem}_{i}{suf}"
            try:
                if move:
                    shutil.move(str(src), str(dest))
                    self.append_log(f"[people][MOVE] {src} -> {dest}")
                else:
                    shutil.copy2(str(src), str(dest))
                    self.append_log(f"[people][COPY] {src} -> {dest}")
            except Exception as e:
                self.append_log(f"[people][ERREUR] {src} -> {dest}: {e}")

        self.refresh_people_list()

    def add_photos_to_selected(self):
        item = self.people_list.currentItem()
        if not item:
            QMessageBox.information(self, "Info", "Sélectionne une personne dans la liste.")
            return
        person = item.data(Qt.UserRole) or item.text().split("—",1)[0].strip()
        files, _ = QFileDialog.getOpenFileNames(
            self, f"Ajouter des photos pour {person}",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif)"
        )
        if not files:
            return
        self.handle_people_drop(person, files)

    # --- Vignettes ---
    def on_person_selection_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]):
        self.thumb_list.clear()
        if not current:
            return
        name = current.data(Qt.UserRole) or current.text().split("—",1)[0].strip()
        people_dir = Path(self.people_edit.text())
        folder = people_dir / name
        if not folder.exists():
            return
        images = list_images(folder)
        # option: limiter pour éviter les freezes si énorme dossier
        max_show = 500
        too_many = len(images) > max_show
        if too_many:
            self.append_log(f"[thumbs] {len(images)} images, affichage des {max_show} premières…")
        for p in images[:max_show]:
            icon = self._make_icon_for_path(p, 128)
            item = QListWidgetItem(icon, p.name)
            item.setData(Qt.UserRole, str(p))
            self.thumb_list.addItem(item)

    def _make_icon_for_path(self, path: Path, thumb_size: int = 128) -> QIcon:
        try:
            # Utilise PIL pour uniformiser (et supporter HEIC)
            im = Image.open(path)
            im = im.convert("RGB")
            im.thumbnail((thumb_size, thumb_size))
            qimage = ImageQt(im)  # PIL -> QImage
            pix = QPixmap.fromImage(qimage)
            return QIcon(pix)
        except Exception:
            # fallback: icône vide
            return QIcon()

    def open_thumbnail_item(self, item: QListWidgetItem):
        p = Path(item.data(Qt.UserRole))
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    # --- Build gallery on demand ---
    def build_gallery_now(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Info", "Arrête d'abord le traitement en cours pour reconstruire manuellement.")
            return
        cfg = self.collect_config()
        core = FaceSorterCore(cfg)
        try:
            self.append_log("[Galerie] Chargement du modèle…")
            core.load_app()
            n = core.rebuild_gallery()
            self.append_log(f"[Galerie] OK ({n} personne(s)).")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Reconstruction échouée : {e}")

    # --- Config ---
    def collect_config(self) -> SorterConfig:
        cfg = SorterConfig(
            people_dir=Path(self.people_edit.text()),
            input_dir=Path(self.input_edit.text()),
            output_dir=Path(self.output_edit.text()),
            min_face_size=self.minface_spin.value(),
            topk=5,
            embed_norm=True,
            match_threshold=self.threshold_spin.value(),
            ambiguous_margin=self.ambig_spin.value(),
            group_window_sec=self.groupwin_spin.value(),
            upscale_factors=(1.5, 2.0),
            duplicate_multi_faces=self.dup_faces_chk.isChecked(),
            ctx_id=-1 if self.ctx_combo.currentIndex() == 0 else 0,
            poll_seconds=self.poll_spin.value(),
            move_instead_copy=self.move_chk.isChecked(),
        )
        # ensure base dirs exist
        ensure_dir(cfg.people_dir)
        ensure_dir(cfg.input_dir)
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.output_dir / cfg.unknown_dirname)
        ensure_dir(cfg.output_dir / cfg.noface_dirname)
        self.save_settings()
        return cfg

    # --- Start/Stop ---
    def start_worker(self):
        if self.worker and self.worker.isRunning():
            return
        cfg = self.collect_config()
        self.worker = SortWorker(cfg)
        self.worker.log.connect(self.append_log)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.gallery_built.connect(lambda n: self.append_log(f"[Galerie] {n} personne(s) dans la galerie."))
        # Barre de progression
        self.worker.progress_set_max.connect(lambda n: self.progress.setMaximum(max(0, n)))
        self.worker.progress_set_value.connect(self.progress.setValue)
        self.worker.progress_set_text.connect(self.progress.setFormat)

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log("[INFO] Démarrage de la surveillance…")

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self.append_log("[INFO] Surveillance arrêtée.")
        # reset barre
        self.progress.setMaximum(1)
        self.progress.setValue(0)
        self.progress.setFormat("En veille")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event: QCloseEvent):
        self.stop_worker()
        event.accept()

# =========================
# Entrée
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

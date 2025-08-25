from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSize, QSettings, QUrl
from PySide6.QtGui import QIcon, QPixmap, QDragEnterEvent, QDropEvent, QDesktopServices
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QAbstractItemView,
)

from ..config import SorterConfig
from ..io.images import list_images, ensure_dir, safe_dirname
from ..worker.sorter_worker import SortWorker


# =========================
# Widget liste People avec drag & drop
# =========================
class PeopleListWidget(QListWidget):
    """Liste des personnes ; accepte le drag & drop de fichiers/dossiers sur une personne."""
    from PySide6.QtCore import Signal

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
        item = self.itemAt(event.position().toPoint()) if hasattr(
            event, "position") else self.itemAt(event.pos())
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
# GUI principale
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceSorter — Tri de photos par personne")
        self.setMinimumSize(QSize(1180, 760))
        self.setWindowIcon(QIcon())  # remplace par une icône si tu en as une

        self.settings = QSettings("hdj-advisor", "FaceSorter")

        # === Colonne gauche : panneau People + vignettes ===
        left = QWidget()
        lv = QVBoxLayout(left)

        self.people_list = PeopleListWidget()
        self.people_count_label = QLabel("0 personne")
        self.people_move_on_drop_chk = QCheckBox(
            "Déplacer lors du dépôt (sinon copie)")

        # zone de création
        create_box = QGroupBox("Créer des dossiers people")
        ch = QHBoxLayout(create_box)
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText(
            "Noms séparés par des virgules (ex: Jonas, Alice)")
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
        grid.addWidget(QLabel("people/ :"), 0, 0)
        grid.addWidget(self.people_edit, 0, 1)
        grid.addWidget(self.b_people, 0, 2)
        grid.addWidget(QLabel("input_photos/ :"), 1, 0)
        grid.addWidget(self.input_edit, 1, 1)
        grid.addWidget(self.b_input, 1, 2)
        grid.addWidget(QLabel("output_photos/ :"), 2, 0)
        grid.addWidget(self.output_edit, 2, 1)
        grid.addWidget(self.b_output, 2, 2)
        v.addWidget(g_paths)

        # Group: paramètres
        g_set = QGroupBox("Paramètres")
        g = QGridLayout(g_set)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.ambig_spin = QDoubleSpinBox()
        self.ambig_spin.setRange(0.0, 0.5)
        self.ambig_spin.setSingleStep(0.01)
        self.minface_spin = QSpinBox()
        self.minface_spin.setRange(8, 256)
        self.groupwin_spin = QSpinBox()
        self.groupwin_spin.setRange(0, 60)
        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(1, 60)
        self.ctx_combo = QComboBox()
        self.ctx_combo.addItems(["CPU (ctx=-1)", "GPU #0 (ctx=0)"])
        self.dup_faces_chk = QCheckBox(
            "Dupliquer la photo pour chaque personne reconnue")
        self.move_chk = QCheckBox("Déplacer (au lieu de copier)")
        g.addWidget(QLabel("Seuil de match (cosine) :"), 0, 0)
        g.addWidget(self.threshold_spin, 0, 1)
        g.addWidget(QLabel("Marge ambiguë :"), 0, 2)
        g.addWidget(self.ambig_spin, 0, 3)
        g.addWidget(QLabel("Taille min visage (px) :"), 1, 0)
        g.addWidget(self.minface_spin, 1, 1)
        g.addWidget(QLabel("Fenêtre rafale (s) :"), 1, 2)
        g.addWidget(self.groupwin_spin, 1, 3)
        g.addWidget(QLabel("Période de poll (s) :"), 2, 0)
        g.addWidget(self.poll_spin, 2, 1)
        g.addWidget(QLabel("Contexte (CPU/GPU) :"), 2, 2)
        g.addWidget(self.ctx_combo, 2, 3)
        g.addWidget(self.dup_faces_chk, 3, 0, 1, 4)
        g.addWidget(self.move_chk, 4, 0, 1, 4)
        v.addWidget(g_set)

        # Controls
        h_ctrl = QHBoxLayout()
        self.build_gallery_btn = QPushButton(
            "(Re)construire la galerie maintenant")
        self.open_people_btn = QPushButton("Ouvrir people/")
        self.open_input_btn = QPushButton("Ouvrir input/")
        self.open_output_btn = QPushButton("Ouvrir output/")
        self.start_btn = QPushButton("Démarrer")
        self.stop_btn = QPushButton("Arrêter")
        self.stop_btn.setEnabled(False)
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
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("En veille")
        v.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        v.addWidget(QLabel("Journal"))
        v.addWidget(self.log, 1)

        # Splitter gauche/droite
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 760])
        self.setCentralWidget(splitter)

        # Signals
        self.b_people.clicked.connect(
            lambda: self.select_dir(self.people_edit))
        self.b_input.clicked.connect(lambda: self.select_dir(self.input_edit))
        self.b_output.clicked.connect(
            lambda: self.select_dir(self.output_edit))
        self.create_people_btn.clicked.connect(self.create_people_folders)
        self.refresh_people_btn.clicked.connect(self.refresh_people_list)
        self.add_photos_btn.clicked.connect(self.add_photos_to_selected)
        self.delete_person_btn.clicked.connect(self.delete_selected_person)
        self.build_gallery_btn.clicked.connect(self.build_gallery_now)
        self.open_people_btn.clicked.connect(
            lambda: self.open_dir(self.people_edit.text()))
        self.open_input_btn.clicked.connect(
            lambda: self.open_dir(self.input_edit.text()))
        self.open_output_btn.clicked.connect(
            lambda: self.open_dir(self.output_edit.text()))
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)

        # Drag & drop depuis la liste
        self.people_list.dropped_files.connect(self.handle_people_drop)
        self.people_list.currentItemChanged.connect(
            self.on_person_selection_changed)
        self.thumb_list.itemDoubleClicked.connect(self.open_thumbnail_item)

        # Defaults & settings
        self.load_settings()
        self.worker: Optional[SortWorker] = None

        # Peupler la liste au démarrage
        self.refresh_people_list()

    # === Helpers généraux ===
    def load_settings(self):
        self.people_edit.setText(self.settings.value(
            "people_dir", str(Path.cwd() / "people")))
        self.input_edit.setText(self.settings.value(
            "input_dir", str(Path.cwd() / "input_photos")))
        self.output_edit.setText(self.settings.value(
            "output_dir", str(Path.cwd() / "output_photos")))
        self.threshold_spin.setValue(
            float(self.settings.value("match_threshold", 0.45)))
        self.ambig_spin.setValue(
            float(self.settings.value("ambiguous_margin", 0.05)))
        self.minface_spin.setValue(
            int(self.settings.value("min_face_size", 24)))
        self.groupwin_spin.setValue(
            int(self.settings.value("group_window_sec", 4)))
        self.poll_spin.setValue(int(self.settings.value("poll_seconds", 5)))
        ctx = int(self.settings.value("ctx_id", -1))
        self.ctx_combo.setCurrentIndex(0 if ctx == -1 else 1)
        self.dup_faces_chk.setChecked(self.settings.value(
            "duplicate_multi_faces", "false") == "true")
        self.move_chk.setChecked(self.settings.value(
            "move_instead_copy", "true") == "true")
        self.people_move_on_drop_chk.setChecked(
            self.settings.value("move_on_drop", "false") == "true")

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
        self.settings.setValue(
            "duplicate_multi_faces", "true" if self.dup_faces_chk.isChecked() else "false")
        self.settings.setValue(
            "move_instead_copy", "true" if self.move_chk.isChecked() else "false")
        self.settings.setValue(
            "move_on_drop", "true" if self.people_move_on_drop_chk.isChecked() else "false")


    def select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(
            self, "Choisir un dossier", edit.text() or str(Path.cwd()))
        if d:
            from ..io.images import list_images
            from pathlib import Path

            p = Path(d).expanduser().resolve()
            edit.setText(str(p))
            self.save_settings()

            # 🔎 Log immédiat du nombre d’images détectées
            n = len(list_images(p))
            label = "people" if edit is self.people_edit else (
                "input" if edit is self.input_edit else "output")
            self.append_log(f"[{label}] {p} — {n} image(s) détectée(s)")

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
            QMessageBox.information(
                self, "Info", "Entrez des noms séparés par des virgules.")
            return
        people_dir = Path(self.people_edit.text())
        ensure_dir(people_dir)
        created = []
        for raw in names_line.split(","):
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
            QMessageBox.information(
                self, "Info", "Sélectionne une personne à supprimer.")
            return
        name = item.data(Qt.UserRole) or item.text().split("—", 1)[0].strip()
        people_dir = Path(self.people_edit.text())
        target = people_dir / name
        if not target.exists() or not target.is_dir():
            QMessageBox.warning(self, "Attention",
                                f"Le dossier {target} n'existe pas.")
            return
        # confirmation
        ret = QMessageBox.question(
            self,
            "Confirmer la suppression",
            f"Supprimer définitivement le dossier '{name}' et tout son contenu ?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        try:
            import shutil

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
            elif p.is_file() and p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
                ".tif",
                ".tiff",
                ".heic",
                ".heif",
            }:
                imgs.append(p)
        return imgs

    def handle_people_drop(self, person_name: str, paths: List[str]):
        """Drag & drop: copie/déplace des fichiers/dossiers dans people/<person_name>/"""
        import shutil

        people_dir = Path(self.people_edit.text())
        target_dir = people_dir / safe_dirname(person_name)
        ensure_dir(target_dir)

        imgs = self._gather_image_paths(paths)
        if not imgs:
            QMessageBox.information(
                self, "Info", "Aucune image reconnue dans le dépôt.")
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
            QMessageBox.information(
                self, "Info", "Sélectionne une personne dans la liste.")
            return
        person = item.data(Qt.UserRole) or item.text().split("—", 1)[0].strip()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            f"Ajouter des photos pour {person}",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif)",
        )
        if not files:
            return
        self.handle_people_drop(person, files)

    # --- Vignettes ---
    def on_person_selection_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]):
        self.thumb_list.clear()
        if not current:
            return
        name = current.data(Qt.UserRole) or current.text().split(
            "—", 1)[0].strip()
        people_dir = Path(self.people_edit.text())
        folder = people_dir / name
        if not folder.exists():
            return
        images = list_images(folder)
        # option: limiter pour éviter les freezes si énorme dossier
        max_show = 500
        if len(images) > max_show:
            self.append_log(
                f"[thumbs] {len(images)} images, affichage des {max_show} premières…")
        for p in images[:max_show]:
            icon = self._make_icon_for_path(p, 128)
            item = QListWidgetItem(icon, p.name)
            item.setData(Qt.UserRole, str(p))
            self.thumb_list.addItem(item)

    def _make_icon_for_path(self, path: Path, thumb_size: int = 128) -> QIcon:
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb_size, thumb_size))
            qimage = ImageQt(im)  # PIL -> QImage
            pix = QPixmap.fromImage(qimage)
            return QIcon(pix)
        except Exception:
            return QIcon()

    def open_thumbnail_item(self, item: QListWidgetItem):
        p = Path(item.data(Qt.UserRole))
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    # --- Build gallery on demand ---
    def build_gallery_now(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "Info", "Arrête d'abord le traitement en cours pour reconstruire manuellement.")
            return
        cfg = self.collect_config()
        try:
            self.append_log("[Galerie] Chargement du modèle…")
            # Utilise un SortWorker éphémère pour réutiliser le même core/service
            worker = SortWorker(cfg)
            worker.core.load_service()
            n = worker.core.rebuild_gallery()
            self.append_log(f"[Galerie] OK ({n} personne(s)).")
        except Exception as e:
            QMessageBox.critical(
                self, "Erreur", f"Reconstruction échouée : {e}")

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
        # Connexions aux signaux (noms depuis worker.sorter_worker)
        self.worker.log_sig.connect(self.append_log)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.gallery_built.connect(lambda n: self.append_log(
            f"[Galerie] {n} personne(s) dans la galerie."))
        # Barre de progression
        self.worker.progress_set_max.connect(
            lambda n: self.progress.setMaximum(max(0, n)))
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

    def closeEvent(self, event):
        self.stop_worker()
        event.accept()


def main() -> None:
    """Entry point for `facesorter-gui` script."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

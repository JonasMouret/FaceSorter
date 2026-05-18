from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSize, QSettings, QUrl, QThread, Signal, QObject
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
    QInputDialog,
    QMessageBox,
    QProgressBar,
    QAbstractItemView,
    QFrame,
    QTabWidget,
)

from facesorter.config import SorterConfig
from facesorter.io.images import list_images, ensure_dir, safe_dirname


# --- Helpers chemins (macOS-friendly) -------------------------------------------------
def _to_local_path(s: str) -> Path:
    """Convertit 'file:///…' en chemin local, gère ~, et renvoie un Path résolu."""
    from urllib.parse import urlparse, unquote
    if not s:
        return Path.cwd().resolve()
    if s.startswith("file://"):
        p = urlparse(s)
        s = unquote(p.path)
    return Path(s).expanduser().resolve()


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
# Chargeur de vignettes async
# =========================
class _ThumbLoader(QObject):
    thumbnail_ready = Signal(str, QIcon)
    finished = Signal()

    def __init__(self, paths: list, size: int = 128):
        super().__init__()
        self._paths = paths
        self._size = size
        self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        for path_str in self._paths:
            if self._cancelled:
                break
            try:
                im = Image.open(path_str).convert("RGB")
                im.thumbnail((self._size, self._size))
                qimage = ImageQt(im)
                pix = QPixmap.fromImage(qimage)
                self.thumbnail_ready.emit(path_str, QIcon(pix))
            except Exception:
                self.thumbnail_ready.emit(path_str, QIcon())
        self.finished.emit()


# =========================
# GUI principale
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceSorter — Tri de photos par personne")
        self.setMinimumSize(QSize(1180, 760))

        self.settings = QSettings("hdj-advisor", "FaceSorter")
        self._thumb_thread: Optional[QThread] = None
        self._thumb_loader: Optional[_ThumbLoader] = None
        self.worker: Optional['SortWorker'] = None
        self.cluster_worker = None

        tabs = QTabWidget()
        tabs.addTab(self._build_guided_tab(), "Tri guidé")
        tabs.addTab(self._build_auto_tab(), "Regroupement automatique")
        self.setCentralWidget(tabs)

        self.load_settings()
        self.refresh_people_list()

    # =========================================================
    # Tab 1 — Tri guidé (mode existant)
    # =========================================================
    def _build_guided_tab(self) -> QWidget:
        # === Colonne gauche : panneau People + vignettes ===
        left = QWidget()
        lv = QVBoxLayout(left)

        self.people_list = PeopleListWidget()
        self.people_count_label = QLabel("0 personne")
        self.people_move_on_drop_chk = QCheckBox("Déplacer lors du dépôt (sinon copier)")

        create_box = QGroupBox("Ajouter des personnes")
        ch = QHBoxLayout(create_box)
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText("Noms séparés par des virgules (ex : Jonas, Alice)")
        self.names_edit.setToolTip("Crée un sous-dossier par nom dans le dossier Personnes.")
        self.create_people_btn = QPushButton("Créer")
        ch.addWidget(self.names_edit, 1)
        ch.addWidget(self.create_people_btn)

        ppl_actions = QHBoxLayout()
        self.refresh_people_btn = QPushButton("↻ Rafraîchir")
        self.add_photos_btn = QPushButton("+ Ajouter des photos…")
        self.rename_person_btn = QPushButton("Renommer…")
        self.delete_person_btn = QPushButton("Supprimer")
        self.delete_person_btn.setObjectName("dangerBtn")
        ppl_actions.addWidget(self.refresh_people_btn)
        ppl_actions.addWidget(self.add_photos_btn)
        ppl_actions.addWidget(self.rename_person_btn)
        ppl_actions.addWidget(self.delete_person_btn)

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

        lv.addWidget(QLabel("Personnes  (déposez des photos directement sur un nom)"))
        lv.addWidget(self.people_list, 1)
        lv.addWidget(self.people_count_label)
        lv.addWidget(self.people_move_on_drop_chk)
        lv.addWidget(create_box)
        lv.addLayout(ppl_actions)
        lv.addWidget(thumbs_box, 2)

        # === Colonne droite : config + logs ===
        right = QWidget()
        v = QVBoxLayout(right)

        g_paths = QGroupBox("Dossiers")
        grid = QGridLayout(g_paths)
        self.people_edit = QLineEdit()
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.b_people = QPushButton("Parcourir…")
        self.b_input = QPushButton("Parcourir…")
        self.b_output = QPushButton("Parcourir…")
        lbl_people = QLabel("Personnes :")
        lbl_people.setToolTip("Dossier contenant un sous-dossier par personne avec ses photos de référence.")
        lbl_input = QLabel("Photos à trier :")
        lbl_input.setToolTip("Dossier source où déposer les photos à analyser.")
        lbl_output = QLabel("Résultat :")
        lbl_output.setToolTip("Dossier destination : les photos seront triées dans des sous-dossiers par personne.")
        grid.addWidget(lbl_people, 0, 0)
        grid.addWidget(self.people_edit, 0, 1)
        grid.addWidget(self.b_people, 0, 2)
        grid.addWidget(lbl_input, 1, 0)
        grid.addWidget(self.input_edit, 1, 1)
        grid.addWidget(self.b_input, 1, 2)
        grid.addWidget(lbl_output, 2, 0)
        grid.addWidget(self.output_edit, 2, 1)
        grid.addWidget(self.b_output, 2, 2)
        v.addWidget(g_paths)

        g_set = QGroupBox("Paramètres de reconnaissance")
        g = QGridLayout(g_set)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setToolTip(
            "Seuil de similarité (cosinus) pour valider une correspondance.\n"
            "Plus élevé = plus strict. Valeur conseillée : 0.40–0.55.\n"
            "Si trop de photos vont en Inconnu, baissez ce seuil."
        )
        self.ambig_spin = QDoubleSpinBox()
        self.ambig_spin.setRange(0.0, 0.5)
        self.ambig_spin.setSingleStep(0.01)
        self.ambig_spin.setToolTip(
            "Zone d'ambiguïté en dessous du seuil : une photo dans cette marge\n"
            "est classée 'Inconnue' au lieu d'être attribuée à tort."
        )
        self.minface_spin = QSpinBox()
        self.minface_spin.setRange(8, 256)
        self.minface_spin.setToolTip(
            "Taille minimale du côté le plus court d'un visage (en pixels).\n"
            "Réduire détecte les petits visages mais ralentit le traitement."
        )
        self.groupwin_spin = QSpinBox()
        self.groupwin_spin.setRange(0, 60)
        self.groupwin_spin.setToolTip(
            "Fenêtre de temps (secondes) pour regrouper des photos prises en rafale.\n"
            "Les photos d'un même groupe partagent le vote majoritaire. 0 = désactivé."
        )
        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(1, 60)
        self.poll_spin.setToolTip(
            "Intervalle (secondes) entre chaque vérification du dossier source."
        )
        self.ctx_combo = QComboBox()
        self.ctx_combo.addItems(["CPU", "GPU #0"])
        self.ctx_combo.setToolTip(
            "Processeur utilisé pour la détection.\n"
            "GPU accélère fortement le traitement si disponible (CUDA)."
        )
        self.dup_faces_chk = QCheckBox("Dupliquer la photo pour chaque personne reconnue")
        self.dup_faces_chk.setToolTip(
            "Si plusieurs visages sont détectés, copie la photo dans chaque dossier concerné."
        )
        self.move_chk = QCheckBox("Déplacer les photos (au lieu de les copier)")
        self.move_chk.setToolTip(
            "Déplacer supprime l'original du dossier source après tri.\n"
            "Copier conserve l'original — utile en mode test."
        )
        lbl_threshold = QLabel("Seuil de reconnaissance :")
        lbl_ambig = QLabel("Marge d'ambiguïté :")
        lbl_minface = QLabel("Taille min. visage (px) :")
        lbl_groupwin = QLabel("Fenêtre rafale (s) :")
        lbl_poll = QLabel("Vérification toutes les (s) :")
        lbl_ctx = QLabel("Processeur :")
        g.addWidget(lbl_threshold, 0, 0)
        g.addWidget(self.threshold_spin, 0, 1)
        g.addWidget(lbl_ambig, 0, 2)
        g.addWidget(self.ambig_spin, 0, 3)
        g.addWidget(lbl_minface, 1, 0)
        g.addWidget(self.minface_spin, 1, 1)
        g.addWidget(lbl_groupwin, 1, 2)
        g.addWidget(self.groupwin_spin, 1, 3)
        g.addWidget(lbl_poll, 2, 0)
        g.addWidget(self.poll_spin, 2, 1)
        g.addWidget(lbl_ctx, 2, 2)
        g.addWidget(self.ctx_combo, 2, 3)
        g.addWidget(self.dup_faces_chk, 3, 0, 1, 4)
        g.addWidget(self.move_chk, 4, 0, 1, 4)
        v.addWidget(g_set)

        h_ctrl = QHBoxLayout()
        self.build_gallery_btn = QPushButton("⟳ Reconstruire la galerie")
        self.build_gallery_btn.setToolTip("Recharge les photos de référence de chaque personne.")
        self.open_people_btn = QPushButton("Ouvrir Personnes")
        self.open_input_btn = QPushButton("Ouvrir Source")
        self.open_output_btn = QPushButton("Ouvrir Résultat")
        self.start_btn = QPushButton("▶  Démarrer")
        self.start_btn.setObjectName("startBtn")
        self.stop_btn = QPushButton("■  Arrêter")
        self.stop_btn.setObjectName("stopBtn")
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

        stats_frame = QFrame()
        stats_frame.setObjectName("statsBar")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(8, 4, 8, 4)
        self.stat_sorted_lbl = QLabel("✔  Triées : 0")
        self.stat_unknown_lbl = QLabel("?  Inconnues : 0")
        self.stat_noface_lbl = QLabel("○  Sans visage : 0")
        for lbl in (self.stat_sorted_lbl, self.stat_unknown_lbl, self.stat_noface_lbl):
            stats_layout.addWidget(lbl)
            stats_layout.addStretch(1)
        v.addWidget(stats_frame)

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

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 760])

        # Signal connections
        self.b_people.clicked.connect(lambda: self.select_dir(self.people_edit))
        self.b_input.clicked.connect(lambda: self.select_dir(self.input_edit))
        self.b_output.clicked.connect(lambda: self.select_dir(self.output_edit))
        self.create_people_btn.clicked.connect(self.create_people_folders)
        self.names_edit.returnPressed.connect(self.create_people_folders)
        self.refresh_people_btn.clicked.connect(self.refresh_people_list)
        self.add_photos_btn.clicked.connect(self.add_photos_to_selected)
        self.rename_person_btn.clicked.connect(self.rename_selected_person)
        self.delete_person_btn.clicked.connect(self.delete_selected_person)
        self.build_gallery_btn.clicked.connect(self.build_gallery_now)
        self.open_people_btn.clicked.connect(lambda: self.open_dir(self.people_edit.text()))
        self.open_input_btn.clicked.connect(lambda: self.open_dir(self.input_edit.text()))
        self.open_output_btn.clicked.connect(lambda: self.open_dir(self.output_edit.text()))
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.people_list.dropped_files.connect(self.handle_people_drop)
        self.people_list.currentItemChanged.connect(self.on_person_selection_changed)
        self.thumb_list.itemDoubleClicked.connect(self.open_thumbnail_item)

        return splitter

    # =========================================================
    # Tab 2 — Regroupement automatique (mode clustering)
    # =========================================================
    def _build_auto_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        info = QLabel(
            "Mode automatique : sélectionnez un dossier de photos, le programme détecte les visages "
            "et regroupe chaque personne dans un dossier Sujet_001, Sujet_002… "
            "Aucune photo de référence n'est nécessaire."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #a6e3a1; padding: 6px 0;")
        v.addWidget(info)

        # Dossiers
        g_dirs = QGroupBox("Dossiers")
        grid = QGridLayout(g_dirs)
        self.auto_input_edit = QLineEdit()
        self.auto_input_edit.setPlaceholderText("Dossier contenant les photos à analyser…")
        self.auto_output_edit = QLineEdit()
        self.auto_output_edit.setPlaceholderText("Dossier où créer les sous-dossiers par sujet…")
        self.auto_b_input = QPushButton("Parcourir…")
        self.auto_b_output = QPushButton("Parcourir…")
        grid.addWidget(QLabel("Photos à analyser :"), 0, 0)
        grid.addWidget(self.auto_input_edit, 0, 1)
        grid.addWidget(self.auto_b_input, 0, 2)
        grid.addWidget(QLabel("Dossier résultat :"), 1, 0)
        grid.addWidget(self.auto_output_edit, 1, 1)
        grid.addWidget(self.auto_b_output, 1, 2)
        v.addWidget(g_dirs)

        # Paramètres
        g_params = QGroupBox("Paramètres")
        g = QGridLayout(g_params)
        self.auto_threshold_spin = QDoubleSpinBox()
        self.auto_threshold_spin.setRange(0.10, 1.0)
        self.auto_threshold_spin.setSingleStep(0.01)
        self.auto_threshold_spin.setValue(0.50)
        self.auto_threshold_spin.setToolTip(
            "Seuil de similarité (cosinus) pour regrouper deux visages dans le même dossier.\n"
            "Valeur recommandée : 0.45–0.55.\n"
            "Trop bas → mélange de personnes différentes. Trop haut → même personne dans plusieurs dossiers."
        )
        self.auto_minface_spin = QSpinBox()
        self.auto_minface_spin.setRange(8, 256)
        self.auto_minface_spin.setValue(24)
        self.auto_minface_spin.setToolTip(
            "Taille minimale (px) du plus petit côté d'un visage détectable.\n"
            "Réduire permet de détecter les visages petits ou éloignés."
        )
        self.auto_ctx_combo = QComboBox()
        self.auto_ctx_combo.addItems(["CPU", "GPU #0"])
        self.auto_ctx_combo.setToolTip("Processeur utilisé. GPU accélère fortement si disponible (CUDA).")
        self.auto_move_chk = QCheckBox("Déplacer les photos (au lieu de les copier)")
        self.auto_move_chk.setToolTip(
            "Déplacer supprime l'original. Copier conserve l'original — recommandé pour un premier essai."
        )
        g.addWidget(QLabel("Seuil de regroupement :"), 0, 0)
        g.addWidget(self.auto_threshold_spin, 0, 1)
        g.addWidget(QLabel("Taille min. visage (px) :"), 0, 2)
        g.addWidget(self.auto_minface_spin, 0, 3)
        g.addWidget(QLabel("Processeur :"), 1, 0)
        g.addWidget(self.auto_ctx_combo, 1, 1)
        g.addWidget(self.auto_move_chk, 1, 2, 1, 2)
        v.addWidget(g_params)

        # Contrôles
        h_ctrl = QHBoxLayout()
        self.auto_open_input_btn = QPushButton("Ouvrir Source")
        self.auto_open_output_btn = QPushButton("Ouvrir Résultat")
        self.auto_start_btn = QPushButton("▶  Analyser et regrouper")
        self.auto_start_btn.setObjectName("startBtn")
        self.auto_stop_btn = QPushButton("■  Arrêter")
        self.auto_stop_btn.setObjectName("stopBtn")
        self.auto_stop_btn.setEnabled(False)
        h_ctrl.addWidget(self.auto_open_input_btn)
        h_ctrl.addWidget(self.auto_open_output_btn)
        h_ctrl.addStretch(1)
        h_ctrl.addWidget(self.auto_start_btn)
        h_ctrl.addWidget(self.auto_stop_btn)
        v.addLayout(h_ctrl)

        # Progression
        self.auto_progress = QProgressBar()
        self.auto_progress.setMinimum(0)
        self.auto_progress.setMaximum(1)
        self.auto_progress.setValue(0)
        self.auto_progress.setTextVisible(True)
        self.auto_progress.setFormat("En attente")
        v.addWidget(self.auto_progress)

        # Stats
        auto_stats_frame = QFrame()
        auto_stats_frame.setObjectName("statsBar")
        asl = QHBoxLayout(auto_stats_frame)
        asl.setContentsMargins(8, 4, 8, 4)
        self.auto_stat_subjects_lbl = QLabel("Sujets détectés : —")
        self.auto_stat_noface_lbl = QLabel("Sans visage : —")
        asl.addWidget(self.auto_stat_subjects_lbl)
        asl.addStretch(1)
        asl.addWidget(self.auto_stat_noface_lbl)
        asl.addStretch(1)
        v.addWidget(auto_stats_frame)

        # Journal
        self.auto_log = QPlainTextEdit()
        self.auto_log.setReadOnly(True)
        v.addWidget(QLabel("Journal"))
        v.addWidget(self.auto_log, 1)

        # Connexions
        self.auto_b_input.clicked.connect(lambda: self._auto_select_dir(self.auto_input_edit))
        self.auto_b_output.clicked.connect(lambda: self._auto_select_dir(self.auto_output_edit))
        self.auto_open_input_btn.clicked.connect(lambda: self.open_dir(self.auto_input_edit.text()))
        self.auto_open_output_btn.clicked.connect(lambda: self.open_dir(self.auto_output_edit.text()))
        self.auto_start_btn.clicked.connect(self._start_auto_cluster)
        self.auto_stop_btn.clicked.connect(self._stop_auto_cluster)

        return w

    # =========================================================
    # Méthodes — onglet Regroupement automatique
    # =========================================================
    def _auto_select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Choisir un dossier", edit.text() or str(Path.cwd()))
        if d:
            p = _to_local_path(d)
            edit.setText(str(p))
            n = len(list_images(p))
            self.auto_append_log(f"[info] {p} — {n} image(s) détectée(s)")

    def auto_append_log(self, text: str):
        self.auto_log.appendPlainText(text)
        self.auto_log.verticalScrollBar().setValue(self.auto_log.verticalScrollBar().maximum())

    def _start_auto_cluster(self):
        from facesorter.worker.cluster_worker import ClusterWorker

        input_dir = _to_local_path(self.auto_input_edit.text())
        output_text = self.auto_output_edit.text().strip()

        if not self.auto_input_edit.text().strip() or not input_dir.exists() or not input_dir.is_dir():
            QMessageBox.warning(self, "Dossier invalide", "Le dossier source est invalide ou vide.")
            return
        if not output_text:
            QMessageBox.warning(self, "Dossier manquant", "Sélectionne un dossier résultat.")
            return

        output_dir = _to_local_path(output_text)
        threshold = self.auto_threshold_spin.value()
        min_face_size = self.auto_minface_spin.value()
        ctx_id = -1 if self.auto_ctx_combo.currentIndex() == 0 else 0
        move = self.auto_move_chk.isChecked()

        self.auto_log.clear()
        self.auto_append_log(f"[INFO] Source : {input_dir}")
        self.auto_append_log(f"[INFO] Résultat : {output_dir}")
        self.auto_append_log(f"[INFO] Seuil : {threshold:.2f}  |  Taille min. visage : {min_face_size}px")

        self.cluster_worker = ClusterWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            threshold=threshold,
            min_face_size=min_face_size,
            upscale_factors=(1.5, 2.0, 3.0),
            ctx_id=ctx_id,
            move=move,
        )
        self.cluster_worker.log_sig.connect(self.auto_append_log)
        self.cluster_worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.cluster_worker.progress_max.connect(lambda n: self.auto_progress.setMaximum(max(1, n)))
        self.cluster_worker.progress_val.connect(self.auto_progress.setValue)
        self.cluster_worker.progress_txt.connect(self.auto_progress.setFormat)
        self.cluster_worker.finished_sig.connect(self._on_cluster_finished)
        self.cluster_worker.finished.connect(lambda: self._set_auto_running(False))

        self._set_auto_running(True)
        self.cluster_worker.start()

    def _stop_auto_cluster(self):
        if self.cluster_worker and self.cluster_worker.isRunning():
            self.cluster_worker.stop()
            self.cluster_worker.wait(3000)
            self.auto_append_log("[INFO] Analyse arrêtée.")
        self._set_auto_running(False)

    def _set_auto_running(self, running: bool):
        self.auto_start_btn.setEnabled(not running)
        self.auto_stop_btn.setEnabled(running)
        if not running:
            current_fmt = self.auto_progress.format()
            if "attente" in current_fmt:
                pass
            else:
                self.auto_progress.setFormat("Terminé")

    def _on_cluster_finished(self, n_subjects: int, n_noface: int):
        self.auto_stat_subjects_lbl.setText(f"Sujets détectés : {n_subjects}")
        self.auto_stat_noface_lbl.setText(f"Sans visage : {n_noface}")
        self.auto_append_log(
            f"[RÉSULTAT] {n_subjects} sujet(s) créé(s), {n_noface} photo(s) sans visage."
        )

    # =========================================================
    # Méthodes partagées — helpers généraux
    # =========================================================
    def load_settings(self):
        self.people_edit.setText(str(_to_local_path(self.settings.value("people_dir", str(Path.cwd() / "people")))))
        self.input_edit.setText(str(_to_local_path(self.settings.value("input_dir", str(Path.cwd() / "input_photos")))))
        self.output_edit.setText(str(_to_local_path(self.settings.value("output_dir", str(Path.cwd() / "output_photos")))))
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

        auto_input = self.settings.value("auto_input_dir", "")
        if auto_input:
            self.auto_input_edit.setText(str(_to_local_path(auto_input)))
        auto_output = self.settings.value("auto_output_dir", "")
        if auto_output:
            self.auto_output_edit.setText(str(_to_local_path(auto_output)))
        self.auto_threshold_spin.setValue(float(self.settings.value("auto_threshold", 0.50)))
        self.auto_minface_spin.setValue(int(self.settings.value("auto_min_face_size", 24)))
        auto_ctx = int(self.settings.value("auto_ctx_id", -1))
        self.auto_ctx_combo.setCurrentIndex(0 if auto_ctx == -1 else 1)
        self.auto_move_chk.setChecked(self.settings.value("auto_move", "false") == "true")

    def save_settings(self):
        self.settings.setValue("people_dir", str(_to_local_path(self.people_edit.text())))
        self.settings.setValue("input_dir", str(_to_local_path(self.input_edit.text())))
        self.settings.setValue("output_dir", str(_to_local_path(self.output_edit.text())))
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

        self.settings.setValue("auto_input_dir", self.auto_input_edit.text())
        self.settings.setValue("auto_output_dir", self.auto_output_edit.text())
        self.settings.setValue("auto_threshold", self.auto_threshold_spin.value())
        self.settings.setValue("auto_min_face_size", self.auto_minface_spin.value())
        auto_ctx_id = -1 if self.auto_ctx_combo.currentIndex() == 0 else 0
        self.settings.setValue("auto_ctx_id", auto_ctx_id)
        self.settings.setValue("auto_move", "true" if self.auto_move_chk.isChecked() else "false")

    def select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Choisir un dossier", edit.text() or str(Path.cwd()))
        if d:
            p = _to_local_path(d)
            edit.setText(str(p))
            self.save_settings()

            n = len(list_images(p))
            label = "people" if edit is self.people_edit else ("input" if edit is self.input_edit else "output")
            self.append_log(f"[{label}] {p} — {n} image(s) détectée(s)")

            if edit is self.people_edit:
                self.refresh_people_list()

    def open_dir(self, path_str: str):
        if not path_str:
            return
        p = _to_local_path(path_str)
        ensure_dir(p)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # =========================================================
    # Méthodes — panneau People (onglet Tri guidé)
    # =========================================================
    def refresh_people_list(self):
        people_dir = _to_local_path(self.people_edit.text())
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
        self.on_person_selection_changed(self.people_list.currentItem(), None)

    def create_people_folders(self):
        names_line = self.names_edit.text().strip()
        if not names_line:
            QMessageBox.information(self, "Info", "Entrez des noms séparés par des virgules.")
            return
        people_dir = _to_local_path(self.people_edit.text())
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
            QMessageBox.information(self, "Info", "Sélectionne une personne à supprimer.")
            return
        name = item.data(Qt.UserRole) or item.text().split("—", 1)[0].strip()
        people_dir = _to_local_path(self.people_edit.text())
        target = people_dir / name
        if not target.exists() or not target.is_dir():
            QMessageBox.warning(self, "Attention", f"Le dossier {target} n'existe pas.")
            return
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
        self.refresh_people_list()
        self.thumb_list.clear()

    def rename_selected_person(self):
        item = self.people_list.currentItem()
        if not item:
            QMessageBox.information(self, "Info", "Sélectionnez une personne à renommer.")
            return
        old_name = item.data(Qt.UserRole) or item.text().split("—", 1)[0].strip()
        people_dir = _to_local_path(self.people_edit.text())
        old_path = people_dir / old_name
        new_raw, ok = QInputDialog.getText(self, "Renommer", f"Nouveau nom pour « {old_name} » :", text=old_name)
        if not ok or not new_raw.strip():
            return
        new_name = safe_dirname(new_raw.strip())
        new_path = people_dir / new_name
        if new_path.exists():
            QMessageBox.warning(self, "Conflit", f"Le dossier « {new_name} » existe déjà.")
            return
        try:
            old_path.rename(new_path)
            self.append_log(f"[Personnes] Renommé : {old_name} → {new_name}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Renommage échoué : {e}")
            return
        self.refresh_people_list()

    def _gather_image_paths(self, paths: List[str]) -> List[Path]:
        imgs: List[Path] = []
        for pstr in paths:
            p = Path(pstr)
            if p.is_dir():
                imgs.extend(list_images(p))
            elif p.is_file() and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
            }:
                imgs.append(p)
        return imgs

    def handle_people_drop(self, person_name: str, paths: List[str]):
        """Drag & drop: copie/déplace des fichiers/dossiers dans people/<person_name>/"""
        import shutil

        people_dir = _to_local_path(self.people_edit.text())
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
        if self._thumb_loader:
            self._thumb_loader.cancel()
        self.thumb_list.clear()
        if not current:
            return
        name = current.data(Qt.UserRole) or current.text().split("—", 1)[0].strip()
        people_dir = _to_local_path(self.people_edit.text())
        folder = people_dir / name
        if not folder.exists():
            return
        images = list_images(folder)
        max_show = 500
        if len(images) > max_show:
            self.append_log(f"[Vignettes] {len(images)} photos, affichage des {max_show} premières.")
        self._thumb_items: dict = {}
        for p in images[:max_show]:
            item = QListWidgetItem(QIcon(), p.name)
            item.setData(Qt.UserRole, str(p))
            self.thumb_list.addItem(item)
            self._thumb_items[str(p)] = item
        self._thumb_loader = _ThumbLoader([str(p) for p in images[:max_show]])
        self._thumb_thread = QThread()
        self._thumb_loader.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_loader.run)
        self._thumb_loader.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumb_loader.finished.connect(self._thumb_thread.quit)
        self._thumb_thread.start()

    def _on_thumbnail_ready(self, path_str: str, icon: QIcon):
        item = getattr(self, "_thumb_items", {}).get(path_str)
        if item:
            item.setIcon(icon)

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
        try:
            self.append_log("[Galerie] Chargement du modèle…")
            from facesorter.worker.sorter_worker import SortWorker
            worker = SortWorker(cfg)
            worker.core.load_service()
            n = worker.core.rebuild_gallery()
            self.append_log(f"[Galerie] OK ({n} personne(s)).")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Reconstruction échouée : {e}")

    # --- Config ---
    def collect_config(self) -> SorterConfig:
        cfg = SorterConfig(
            people_dir=_to_local_path(self.people_edit.text()),
            input_dir=_to_local_path(self.input_edit.text()),
            output_dir=_to_local_path(self.output_edit.text()),
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
        ensure_dir(cfg.people_dir)
        ensure_dir(cfg.input_dir)
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.output_dir / cfg.unknown_dirname)
        ensure_dir(cfg.output_dir / cfg.noface_dirname)
        self.append_log(f"[CFG] people={cfg.people_dir}")
        self.append_log(f"[CFG] input={cfg.input_dir}")
        self.append_log(f"[CFG] output={cfg.output_dir}")
        self.save_settings()
        return cfg

    # --- Start/Stop ---
    def start_worker(self):
        if self.worker and self.worker.isRunning():
            return
        cfg = self.collect_config()
        from facesorter.worker.sorter_worker import SortWorker
        self.worker = SortWorker(cfg)
        self.worker.log_sig.connect(self.append_log)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.gallery_built.connect(lambda n: self.append_log(f"[Galerie] {n} personne(s) chargée(s)."))
        self.worker.stats_updated.connect(self._update_stats)
        self.worker.progress_set_max.connect(lambda n: self.progress.setMaximum(max(0, n)))
        self.worker.progress_set_value.connect(self.progress.setValue)
        self.worker.progress_set_text.connect(self.progress.setFormat)

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log("[INFO] Démarrage de la surveillance…")

    def _update_stats(self, sorted_n: int, unknown_n: int, noface_n: int):
        self.stat_sorted_lbl.setText(f"✔  Triées : {sorted_n}")
        self.stat_unknown_lbl.setText(f"?  Inconnues : {unknown_n}")
        self.stat_noface_lbl.setText(f"○  Sans visage : {noface_n}")

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self.append_log("[INFO] Surveillance arrêtée.")
        self.progress.setMaximum(1)
        self.progress.setValue(0)
        self.progress.setFormat("En veille")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        self.stop_worker()
        self._stop_auto_cluster()
        event.accept()


_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "SF Pro Display", system-ui, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 6px 18px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}
QTabBar::tab:hover:!selected { background-color: #45475a; }
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 6px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #89b4fa;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 12px;
}
QPushButton:hover  { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { color: #585b70; border-color: #313244; }
QPushButton#startBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#startBtn:hover   { background-color: #94d49e; }
QPushButton#startBtn:disabled { background-color: #313244; color: #585b70; border: 1px solid #45475a; font-weight: normal; }
QPushButton#stopBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#stopBtn:hover    { background-color: #eb7a96; }
QPushButton#stopBtn:disabled { background-color: #313244; color: #585b70; border: 1px solid #45475a; font-weight: normal; }
QPushButton#dangerBtn {
    color: #f38ba8;
    border-color: #f38ba8;
}
QPushButton#dangerBtn:hover { background-color: #3d1f27; }
QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-radius: 3px;
}
QListWidget::item:hover:!selected { background-color: #313244; }
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #181825;
    text-align: center;
    color: #cdd6f4;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QSplitter::handle { background-color: #313244; width: 2px; }
QScrollBar:vertical {
    background: #181825;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #313244;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    border: 1px solid #45475a;
}
QFrame#statsBar {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
}
QLabel { color: #cdd6f4; }
QStatusBar { background-color: #181825; color: #6c7086; }
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
"""


def _app_icon() -> QIcon:
    """Load the app icon from the resources package, compatible with PyInstaller."""
    import os
    import sys

    candidates = []
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller bundle
        candidates.append(os.path.join(sys._MEIPASS, "facesorter", "resources", "icon.png"))
    # Development / installed package
    candidates.append(str(Path(__file__).parent.parent / "resources" / "icon.png"))

    for p in candidates:
        if os.path.exists(p):
            return QIcon(p)
    return QIcon()


def main() -> None:
    """Entry point for `facesorter-gui` script."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setWindowIcon(_app_icon())
    app.setStyleSheet(_STYLESHEET)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

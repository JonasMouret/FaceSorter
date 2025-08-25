from __future__ import annotations
import time, threading, logging
from pathlib import Path
from typing import Optional, Callable, List
from PySide6.QtCore import QThread, Signal
from ..config import SorterConfig
from ..core.sorter import FaceSorterCore, dir_signature
from ..io.images import list_images

log = logging.getLogger(__name__)

class SortWorker(QThread):
    log_sig = Signal(str)
    status = Signal(str)
    gallery_built = Signal(int)
    progress_set_max = Signal(int)
    progress_set_value = Signal(int)
    progress_set_text = Signal(str)

    def __init__(self, cfg: SorterConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.core = FaceSorterCore(cfg)
        self._stop = threading.Event()
        self._people_sig: Optional[str] = None

    def stop(self): self._stop.set()

    def run(self):
        try:
            self.status.emit("Chargement du modèle…")
            self.core.load_service()
            self.status.emit("Modèle prêt.")
        except Exception as e:
            self.log_sig.emit(f"[ERREUR] InsightFace: {e}")
            self.status.emit("Erreur modèle — thread stoppé.")
            return

        self.log_sig.emit("[Galerie] Initialisation…")
        self.progress_set_max.emit(0); self.progress_set_text.emit("Construction de la galerie…")
        n_init = self.core.rebuild_gallery()
        self.gallery_built.emit(n_init)
        self._people_sig = dir_signature(self.cfg.people_dir)
        self.progress_set_max.emit(1); self.progress_set_value.emit(0); self.progress_set_text.emit("En veille")

        while not self._stop.is_set():
            try:
                sig = dir_signature(self.cfg.people_dir)
                if sig != self._people_sig:
                    self.log_sig.emit("[Galerie] Changement détecté → reconstruction…")
                    self.progress_set_max.emit(0); self.progress_set_text.emit("Mise à jour de la galerie…")
                    n = self.core.rebuild_gallery()
                    self.gallery_built.emit(n)
                    self._people_sig = sig
                    self.progress_set_max.emit(1); self.progress_set_value.emit(0); self.progress_set_text.emit("En veille")

                if not self.core.gallery.vectors:
                    time.sleep(self.cfg.poll_seconds); continue

                files_now = list_images(self.cfg.input_dir)
                self.log_sig.emit(f"[INFO] Fichiers détectés: {len(files_now)}")

                total = len(files_now)
                if total > 0:
                    self.progress_set_max.emit(total); self.progress_set_value.emit(0)
                    self.progress_set_text.emit(f"Traitement de {total} fichier(s)…")
                else:
                    self.progress_set_max.emit(1); self.progress_set_value.emit(0); self.progress_set_text.emit("En veille")

                def _progress_cb(i, n): self.progress_set_value.emit(i)
                count = self.core.process_files(files_now, self.log_sig.emit, _progress_cb)

                if count == 0:
                    self.progress_set_text.emit("En veille"); time.sleep(self.cfg.poll_seconds)
                else:
                    self.progress_set_text.emit("Terminé")
            except Exception as e:
                self.log_sig.emit(f"[ERREUR] Boucle: {e}")
                time.sleep(self.cfg.poll_seconds)

        self.status.emit("Thread arrêté.")

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from facesorter._version import __version__

_GITHUB_API = "https://api.github.com/repos/JonasMouret/FaceSorter/releases/latest"
_TIMEOUT = 8  # secondes


def _asset_name() -> str:
    """Nom du ZIP attendu pour cette plateforme/architecture."""
    if platform.system() == "Darwin":
        arch = "arm64" if platform.machine() == "arm64" else "x86_64"
        return f"FaceSorter-macOS-{arch}.zip"
    if platform.system() == "Windows":
        return "FaceSorter-Windows-x86_64.zip"
    return ""


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.lstrip("v").split("."))
    except ValueError:
        return (0,)


# ---------------------------------------------------------------------------
# Checker — vérifie GitHub au démarrage
# ---------------------------------------------------------------------------

class UpdateChecker(QThread):
    update_available = Signal(str, str)  # new_version, download_url

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                _GITHUB_API,
                headers={"User-Agent": f"FaceSorter/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                data = json.loads(r.read())

            latest = data.get("tag_name", "").lstrip("v")
            if not latest:
                return
            if _version_tuple(latest) <= _version_tuple(__version__):
                return

            asset_name = _asset_name()
            if not asset_name:
                return

            url = next(
                (a["browser_download_url"]
                 for a in data.get("assets", [])
                 if a["name"] == asset_name),
                "",
            )
            if url:
                self.update_available.emit(latest, url)
        except Exception:
            pass  # échec silencieux — pas de réseau, etc.


# ---------------------------------------------------------------------------
# Installer — télécharge et installe la mise à jour
# ---------------------------------------------------------------------------

class UpdateInstaller(QThread):
    progress = Signal(int)        # 0-100
    done = Signal(bool, str)      # success, message

    def __init__(self, url: str, version: str, parent=None):
        super().__init__(parent)
        self.url = url
        self.version = version

    def run(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="facesorter_update_"))
        zip_path = tmp / "update.zip"
        try:
            self._download(zip_path)
            self.progress.emit(90)
            self._install(zip_path, tmp)
            self.done.emit(True, "")
        except PermissionError:
            # Fallback : télécharger dans ~/Downloads
            try:
                dest = self._save_to_downloads(zip_path)
                self.done.emit(
                    False,
                    f"Droits insuffisants pour mettre à jour automatiquement.\n"
                    f"Le ZIP a été copié dans :\n{dest}\n\n"
                    f"Fermez l'application, extrayez le ZIP et remplacez l'ancienne version.",
                )
            except Exception as e:
                self.done.emit(False, f"Téléchargement échoué : {e}")
        except Exception as e:
            self.done.emit(False, str(e))
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass

    # ------------------------------------------------------------------

    def _download(self, dest: Path) -> None:
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": f"FaceSorter/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            received = 0
            with open(dest, "wb") as f:
                while chunk := r.read(65536):
                    f.write(chunk)
                    received += len(chunk)
                    if total:
                        self.progress.emit(int(received * 88 / total))

    def _save_to_downloads(self, zip_path: Path) -> Path:
        downloads = Path.home() / "Downloads"
        downloads.mkdir(exist_ok=True)
        dest = downloads / f"FaceSorter-v{self.version}.zip"
        shutil.copy2(zip_path, dest)
        return dest

    def _install(self, zip_path: Path, tmp: Path) -> None:
        if not getattr(sys, "frozen", False):
            # Mode développement : on télécharge seulement dans ~/Downloads
            dest = self._save_to_downloads(zip_path)
            raise PermissionError(
                f"Mode développement : mise à jour automatique désactivée.\n"
                f"ZIP disponible dans : {dest}"
            )

        if platform.system() == "Darwin":
            self._install_macos(zip_path, tmp)
        elif platform.system() == "Windows":
            self._install_windows(zip_path, tmp)
        else:
            raise NotImplementedError("Auto-update non supporté sur cette plateforme.")

    def _install_macos(self, zip_path: Path, tmp: Path) -> None:
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        subprocess.run(
            ["ditto", "-x", "-k", str(zip_path), str(extract_dir)],
            check=True,
        )

        apps = list(extract_dir.rglob("*.app"))
        if not apps:
            raise RuntimeError("FaceSorter.app introuvable dans l'archive.")
        new_app = apps[0]

        # Chemin de l'app courante : .../FaceSorter.app/Contents/MacOS/FaceSorter
        current_exe = Path(sys.executable)
        current_app = current_exe.parent.parent.parent

        script = (
            f"#!/bin/bash\n"
            f"sleep 2\n"
            f"rm -rf '{current_app}'\n"
            f"cp -r '{new_app}' '{current_app}'\n"
            f"open '{current_app}'\n"
        )
        script_path = tmp / "do_update.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        subprocess.Popen([str(script_path)], close_fds=True)

    def _install_windows(self, zip_path: Path, tmp: Path) -> None:
        import zipfile

        extract_dir = tmp / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        new_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if not new_dirs:
            raise RuntimeError("Dossier FaceSorter introuvable dans l'archive.")
        new_dir = new_dirs[0]

        current_exe = Path(sys.executable)
        current_dir = current_exe.parent

        bat = (
            "@echo off\r\n"
            "timeout /t 2 /nobreak > nul\r\n"
            f'xcopy /s /e /y "{new_dir}\\*" "{current_dir}\\"\r\n'
            f'start "" "{current_dir}\\FaceSorter.exe"\r\n'
            "del \"%~f0\"\r\n"
        )
        bat_path = tmp / "do_update.bat"
        bat_path.write_text(bat, encoding="utf-8")
        subprocess.Popen(
            ["cmd.exe", "/c", str(bat_path)],
            close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )

# app.py
import sys
from pathlib import Path

# Ajoute src/ au sys.path pour un lancement direct
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tolérant aux deux noms de fichier
try:
    from facesorter.gui.main_windows import main
except ModuleNotFoundError:
    from facesorter.gui.main_window import main  # fallback

if __name__ == "__main__":
    main()

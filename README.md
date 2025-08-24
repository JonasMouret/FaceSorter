# FaceSorter

**FaceSorter** est une application graphique multiplateforme (**macOS, Windows, Linux**) permettant de **trier/déplacer automatiquement des photos par personne** à l’aide de [InsightFace](https://github.com/deepinsight/insightface).  

---

## ✨ Fonctionnalités

- Interface graphique (Qt / PySide6)
- Choix des dossiers `people/`, `input_photos/`, `output_photos/`
- Lister, créer, supprimer des personnes (`people/<Nom>`)
- Glisser-déposer des photos/dossiers directement sur une personne
- Aperçu en vignettes des photos d’un dossier (double-clic = ouverture)
- Paramétrage : seuils de similarité, taille visage min., fenêtre de rafale, duplication multi-visages…
- Traitement continu (poll toutes les N secondes) avec barre de progression
- Déplacement ou copie des photos triées vers `output_photos/<Nom>/`
- Reconstruction automatique de la galerie quand `people/` change
- **Mode hors-ligne** : les modèles InsightFace (`buffalo_l`) sont embarqués dans l’application

---

## 📦 Dépendances

Python ≥ 3.10  

Bibliothèques Python :
- `PySide6`
- `insightface`
- `onnxruntime` (ou `onnxruntime-gpu` si GPU NVIDIA est disponible)
- `opencv-python`
- `pillow`
- `pillow-heif`
- `numpy`

Librairies natives :
- **Linux** : `libheif` (`sudo apt install libheif1 libheif-dev`)
- **macOS** : via Homebrew → `brew install libheif`

👉 Toutes les dépendances Python sont listées dans [`requirements.txt`](requirements.txt).

---

## 🚀 Installation (mode développement)

Clone le repo et installe les dépendances :

```bash
git clone https://github.com/<ton_user>/<ton_repo>.git
cd <ton_repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

## ▶️ Lancer l’application

```bash
python sort_photos_by_person.py
```

---

## 🖼 Utilisation

1. Lance l’app → configure les chemins `people/`, `input_photos/`, `output_photos/`.
2. Ajoute des dossiers pour chaque personne dans `people/` (via l’interface ou drag & drop).
3. Mets quelques photos de référence de chaque personne dans son dossier.
4. Place tes photos brutes dans `input_photos/`.
5. Clique sur **Démarrer** → les photos sont classées automatiquement dans `output_photos/<Nom>/`.

---

## 🔒 Mode hors-ligne

Les modèles InsightFace (`buffalo_l`) sont intégrés dans le dossier `insightface_home/models/buffalo_l`.
Au runtime, le script force `INSIGHTFACE_HOME` vers ce dossier embarqué → aucun téléchargement n’est requis.

---

## 🛠 Compilation avec PyInstaller

### Linux / Windows (local)

```bash
pip install pyinstaller
pyinstaller FaceSorter.spec
```

Résultat :

* **Linux** → `dist/FaceSorter/`
* **Windows** → `dist/FaceSorter.exe`

### macOS (via GitHub Actions)

Depuis Ubuntu, tu ne peux pas générer directement une app macOS.
👉 Utilise un workflow **GitHub Actions** avec runner macOS.

Exemple : `.github/workflows/macos-build.yml` est fourni pour construire **FaceSorter.app** et publier un ZIP.

Artifacts générés → `FaceSorter-macOS.zip` contenant l’app autonome.

---

## ⚠️ Notes importantes

* Première ouverture macOS : clic droit → **Ouvrir** (app non signée).
* Pour une distribution large, ajoute une étape de **signature et notarisation** Apple Developer.
* Pour de très gros dossiers de photos, l’affichage des vignettes est limité (par défaut 500 images max affichées).

---

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).  
© 2025 Jonas Mouret

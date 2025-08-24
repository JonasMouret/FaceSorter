# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.  
Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/)  
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

---

## [0.1.0] - 2025-08-24
### ✨ Ajouté
- Première version publique de **FaceSorter** 🎉
- Interface graphique (PySide6) multiplateforme (Linux, Windows, macOS).
- Création et suppression de personnes dans `people/`.
- Glisser-déposer de photos ou dossiers directement sur une personne.
- Aperçu en vignettes des photos par personne (double-clic = ouverture).
- Paramétrage complet (seuils de similarité, taille min. visage, rafale, duplication multi-visages).
- Traitement continu avec barre de progression.
- Déplacement ou copie des photos dans `output_photos/<Nom>/`.
- Support des formats HEIC/HEIF (via `pillow-heif`).

### 🐛 Problèmes connus
- L’affichage est limité à **500 vignettes max** par dossier pour éviter les freezes.
- Le premier lancement peut prendre du temps (téléchargement et initialisation du modèle InsightFace).
- Sur macOS, l’application n’est pas encore signée/notarisée (clic droit → *Ouvrir* nécessaire).

---

# Détection Automatique de Lésions du Genou par IRM avec Deep Learning



## 📌 Table des Matières
1. [Objectifs du Projet](#-objectifs-du-projet)
2. [Jeu de Données](#-jeu-de-données)
3. [Prétraitement](#-prétraitement)
4. [Architecture du Modèle](#-architecture-du-modèle)
5. [Résultats](#-résultats)
6. [Déploiement](#-déploiement)
7. [Perspectives](#-perspectives)

## 🎯 Objectifs du Projet

### Contexte Médical
Les lésions ligamentaires du genou affectent des millions de patients annuellement. Ce projet vise à automatiser le diagnostic à partir d'IRM pour :

- **Réduire le temps d'analyse** de 30 min à quelques secondes
- **Standardiser les diagnostics** (variabilité inter-radiologues : ±23%)
- **Détecter les lésions subtiles** souvent manquées

### Pathologies Ciblées
- Déchirures du LCA (Ligament Croisé Antérieur)
- Lésions méniscales
- Anomalies générales du genou

## 📊 Jeu de Données

**Source** : MRNet-v1.0 (Stanford Medicine)

| Caractéristique          | Détails                          |
|--------------------------|----------------------------------|
| Examens d'entraînement   | 1,130 IRM annotées              |
| Examens de validation    | 120 IRM                         |
| Format                   | .npy (NumPy arrays)             |
| Résolution               | 256×256 pixels (20-40 tranches) |
| Annotations              | Binaires (0=normal, 1=anormal)  |

**📊 Compréhension des Données** :
 Structure des données
Images IRM au format .npy (tableaux NumPy)
Chaque IRM contient 20 à 40 tranches en niveaux de gris
Résolution originale : 256×256 pixels
 **🔢 Annotations binaires**
Anomalie générale : 0 = normal, 1 = anormal
Lésion du LCA : 0/1
Lésion méniscale : 0/1
**📑 Métadonnées**
Fichiers CSV contenant :
Identifiant de l’examen (0000.npy)
Label associé (0 ou 1)
Fichiers : train-abnormal.csv, train-acl.csv, train-meniscus.csv
**👁️‍🗨️ Aperçu des données**
Nombre d’IRM d’entraînement : 1130
Exemple de forme : (20, 256, 256)
Après prétraitement : (1130, 3, 224, 224)
## 🛠 Préparation des Données
**🧼 Prétraitement des images**
Conversion des fichiers .npy en tableaux NumPy
Normalisation des pixels (valeurs entre 0 et 1)
Sélection de 3 tranches centrales par IRM
Redimensionnement à 224x224
Conversion au format float32
**🔁 Augmentation de données (phase d'entraînement uniquement)**
Rotation aléatoire (±15°)
Zoom (90-110%)
Retournement horizontal (50 % de chance)
**🏷 Préparation des labels**
Réplication des labels pour les 3 tranches
Conversion en one-hot encoding pour l'entraînement multi-classes
Dictionnaire {nom_fichier: label}
## 🧠 Modélisation
**🧬 CNN Personnalisé**
| Couche                  | Détails                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| Entrée                  | Image (224 × 224 × 1)                                                   |
| Conv2D                  | 32 filtres, taille 3×3, activation ReLU                                 |
| MaxPooling2D            | Taille 2×2                                                              |
| BatchNormalization      | -                                                                       |
| Conv2D                  | 64 filtres, taille 3×3, activation ReLU                                 |
| MaxPooling2D            | Taille 2×2                                                              |
| BatchNormalization      | -                                                                       |
| Conv2D                  | 128 filtres, taille 3×3, activation ReLU                                |
| MaxPooling2D            | Taille 2×2                                                              |
| BatchNormalization      | -                                                                       |
| Flatten                 | Mise à plat de la sortie des convolutions                              |
| Dense                   | 256 neurones, activation ReLU                                           |
| Dropout                 | Taux de dropout : 0.6                                                   |
| Dense (Sortie)          | 1 neurone, activation Sigmoïde (classification binaire)                |

**Résultats**
Exactitude finale : ~82.82 %
Perte finale : ~0.4483
Temps par epoch : ~270 secondes
# 5. 🧪 Évaluation du Modèle
**📊 Métriques principales**
Exactitude (Accuracy) : 85.2 %
Précision : 83.7 %
Rappel (Recall) : 86.9 %
## 🚀 Déploiement
Application avec Streamlit
L’application déployée permet :
📤 Téléversement d’IRM
🤖 Analyse automatique via 3 modèles dédiés :
Détection des déchirures du LCA
Détection des lésions méniscales
Détection des anomalies générales
📊 Affichage des probabilités pour chaque diagnostic

**lien de la presentation :**
https://www.canva.com/design/DAGlYnUA0vA/qSmh8j2uv9iusAls2yPYTA/edit?utm_content=DAGlYnUA0vA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


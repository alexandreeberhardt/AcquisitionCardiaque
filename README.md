# ECG Real-Time Acquisition & Analysis (NI-DAQ + PyQt)

Application Python permettant l’acquisition, la visualisation et l’analyse d’un signal ECG en temps réel.

- Acquisition NI-DAQ (USB-6000) ou lecture d’un fichier
- Interface PyQt5 + PyQtGraph
- Filtres numériques (passe-bas / passe-haut / moyenne glissante)
- Détection des pics R (Pan–Tompkins modifié + recentrage)
- Calcul du BPM
- Export CSV (mode live) et capture d’écran PNG

---

## Aperçu

Deux modes sont disponibles :

- **Acquisition en direct** : via NI USB-6000
- **Lecture fichier** : relecture d’un enregistrement (si pas de matériel ou pour simple relecture)

---

## Fonctionnalités

### Acquisition
- Acquisition continue (par défaut 1000 Hz, paramétrable dans le code)
- Changement dynamique du canal AI
- Buffer mémoire borné via `deque`

### Affichage
- **Scrolling** : fenêtre glissante (sur les dernieres secondes)
- **Lecture simple** : centrage autour du dernier pic R détecté
- Curseurs interactifs : Δt (2 curseurs verticaux) + amplitude (curseur horizontal)

### Traitement du signal
- Inversion des électrodes à l’affichage
- Filtre passe-bas Butterworth
- Filtre passe-haut Butterworth
- Moyenne glissante
- Paramètres ajustables en temps réel (à l’affichage)

### Détection QRS / pics R
- Pan–Tompkins (version modifiée)
- Recentrage des pics sur le signal brut (`refine_peaks_on_raw_ecg`)
- BPM via moyenne des intervalles RR (avec filtrage des RR aberrants)

### Export
- Export **CSV** (mode live uniquement)
- Capture d’écran **PNG**
- Vue d’ensemble en mode lecture fichier (overview)

---

## Structure du projet

```text
.
├── main.py
├── ecg_detectors_modified.py
├── data/
│   └── fichier_test.txt
├── screenshots/
├── requirements.txt
├── LICENSE
└── README.md
```
## Installation 

git clone https://github.com/yass2212/AcquisitionCardiaque.git
cd AcquisitionCardiaque

### Dépendence 

pip install -r requirements.txt

### Lancement 

python main.py

## Utilisation

### Acquisition en direct 

- Brancher le boîtier NI USB-6000
- Cliquer sur Acquisition en direct
- Sélectionner le canal AI
- (Optionnel) Activer inversion / filtres
- (Optionnel) Cliquer sur Enregistrer pour exporter un CSV

### Lecture fichier 

- Cliquer sur Lecture fichier
- Sélectionner un fichier de test
- La vue d’ensemble s’affiche et un curseur indique l’avancement

Le fichier attend deux colonnes comme suit avec le temps en seconde et la tension en volt : 
Tps (s)    V1 (V)
0.000      0.012
0.001      0.015
...

## Référence
Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering.

Code réalisé dans le cadre d'une refonte d'un TP pour l'Université de Technologie de Compiègne. Le code a été réalisé par Justine Vérité, étudiante de l'UTC et moi-même, Yassine Ben Ammar, étudiant à l'UTC, supervisé par les Professeurs Dan Istrate et Jeremy Laforet de l'Université de Technologie de Compiègne. 



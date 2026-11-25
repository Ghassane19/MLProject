# Classification d’objets routiers — Livrable 2

Projet PACTE — 5 000 images RGB PNG de 128 × 128 pixels, réparties en cinq classes :
`bus`, `car`, `cyclist`, `motorcycle` et `pedestrian`.
Le jeu initial contient 800 images d’entraînement et 200 images test par classe.

## Les deux notebooks à utiliser

- [Nettoyage_des_donnees.ipynb](notebooks/Nettoyage_des_donnees.ipynb) : contrôle de qualité initial, analyse des images et nettoyage en mémoire.
- [Livrable_2_Preparation_Selection_Entrainement.ipynb](notebooks/Livrable_2_Preparation_Selection_Entrainement.ipynb) : étude comparative étendue, préparation, sélection des variables, apprentissage, analyse critique et pistes d’amélioration.

Le notebook du livrable 2 recharge les résultats enregistrés d’une exécution complète
du protocole `expanded-v2`. Mettre `RUN_EXPERIMENTS = True` pour refaire tous les calculs.
Le chargement contrôle la version du protocole et l’état de l’exécution afin de ne pas
présenter des résultats partiels comme un bilan complet. Les fichiers images originaux
ne sont jamais modifiés.

## Ce que fait l’étude

Chaque image est convertie en **1 860 caractéristiques** : 1 764 HOG pour les contours et
les formes, et 96 valeurs d’histogrammes HSV pour les couleurs. L’audit vérifie les
fichiers, les empreintes de pixels, les étiquettes et les groupes de provenance.
La validation est extraite du train sans couper les groupes apparentés.

L’étude compare **10 modèles couvrant plusieurs familles d’algorithmes** :

| Modèle | Principe |
|---|---|
| SVM RBF | Sépare les classes avec des frontières non linéaires. |
| SGD, perte hinge | Apprend un séparateur linéaire par descente de gradient stochastique. |
| Régression logistique | Apprend un modèle linéaire multiclasse. |
| Random Forest | Combine les votes de nombreux arbres. |
| Extra Trees | Combine des arbres avec des seuils de séparation aléatoires. |
| k-NN | Compare l’image à ses voisins dans l’espace des caractéristiques. |
| Arbre de décision | Apprend une succession de règles de séparation. |
| Gaussian Naive Bayes | Modélise les caractéristiques par classe avec une hypothèse d’indépendance. |
| HistGradientBoosting | Construit des arbres successifs pour corriger les erreurs. |
| MLP | Apprend un réseau de neurones dense sur les descripteurs HOG/HSV. |

Le classifieur SGD utilise la perte hinge, apparentée au SVM linéaire, avec un
apprentissage adapté au coût de calcul des descripteurs. La comparaison porte sur
dix modèles couvrant plusieurs approches ; certains partagent des principes communs.

Le protocole comporte plusieurs étapes :

1. **Présélection : 10 modèles × 3 sélections × 2 stratégies = 60 configurations.**
   Les sélections sont les variables complètes non constantes, ANOVA 50 % et importance
   Random Forest 50 %. Le train interne est utilisé avec ou sans correction du déséquilibre.
   Deux références simples — classe majoritaire et tirage aléatoire stratifié — servent de repères.
2. **Réglage ciblé :** les trois modèles distincts les mieux placés en validation sont
   réglés sur de petites grilles d’hyperparamètres avec une validation croisée à trois plis
   `StratifiedGroupKFold`. Les transformations et la sélection sont réajustées sur le train
   de chaque pli. Le nombre exact d’ajustements et les paramètres sont publiés.
3. **Décision :** le candidat au meilleur F1 macro moyen en validation croisée est retenu
   parmi ces trois modèles présélectionnés. Les sept autres modèles disposent d’une
   comparaison sur la validation initiale, sans le même réglage en validation croisée.
4. **Analyses complémentaires :** comparaison HOG seul / HSV seul / combinaison avec le
   modèle et les paramètres retenus, sans sélection de variables ; courbe d’apprentissage
   sur des sous-ensembles emboîtés à groupes indivisibles ; analyse du coût de calcul,
   du surapprentissage et des avertissements de convergence.
5. **Évaluation finale :** le meilleur candidat de chaque modèle est réajusté sur les
   4 000 images de développement. Les dix finalistes sont évalués sur le test existant,
   avec résultats par classe, matrices de confusion, erreurs visuelles et scores par source.
   Le classement test ne modifie pas le gagnant déjà choisi.
6. **Incertitude :** intervalles percentile à 95 % issus de 1 000 rééchantillonnages
   de groupes test entiers. Les cinq classes restent définies dans les scores,
   y compris pour les tirages où une classe manque ; leur fréquence est publiée.

La correction du déséquilibre utilise des poids de classes pour SVM, SGD, régression
logistique, Random Forest, Extra Trees et arbre de décision. Elle utilise des
duplications de lignes du train en mémoire pour k-NN, GaussianNB,
HistGradientBoosting et MLP. Pour HistGradientBoosting, le suréchantillonnage évite
le surcoût observé du calcul pondéré des histogrammes ; ses poids de classes
restent désactivés. La validation et le test ne sont jamais rééquilibrés.
Le jeu initial est équilibré, mais la séparation par groupes crée un déséquilibre
modéré dans le train interne.

## Résultats calculés

Le [notebook du livrable 2](notebooks/Livrable_2_Preparation_Selection_Entrainement.ipynb)
présente les tableaux et graphiques de l’exécution. La synthèse est générée automatiquement
dans [RESULTATS.md](backend/models/livrable2/RESULTATS.md), et les mesures détaillées
dans `evaluation_metrics.json`. Consulter ces fichiers après une exécution complète
pour connaître le modèle retenu, ses paramètres et les scores réellement obtenus.

Les scores ne sont pas recopiés manuellement ici : le notebook et la synthèse se fondent
sur les mêmes fichiers de résultats, ce qui évite de mélanger les anciens essais et le
protocole étendu.

## Installation et exécution

Depuis la racine du projet :

```powershell
python -m pip install -r backend/requirements.txt
python -m backend.livrable2
```

La commande `python -m backend.ml_pipeline` lance le même protocole.
Pour prédire la classe d’une image avec le modèle retenu :

```powershell
python -m backend.ml_pipeline --predict chemin/vers/image.png
```

La prédiction utilise l’artefact `selected_pipeline.joblib`. Selon le modèle retenu,
les probabilités peuvent être disponibles ou les champs `confidence` et
`probabilities` peuvent valoir `null`. Une probabilité disponible n’est pas une
garantie de fiabilité et n’a pas été calibrée pour un usage routier réel.

Pour exécuter les vérifications :

```powershell
python -m unittest discover -s tests -v
```

## Rôle des fichiers conservés

| Fichier ou dossier | Utilité |
|---|---|
| `notebooks/` | Les deux notebooks pédagogiques avec leurs résultats. |
| `dataset/train/` et `dataset/test/` | Les images réelles des expériences. |
| `dataset/sources.csv` | Provenance, étiquettes, empreintes et groupes ; indispensable à l’audit. |
| `dataset/PROJECT_SPECIFICATION.md` | Cahier des charges d’origine et périmètre prévu du projet. |
| `dataset/demo_scenes/` | Scènes réservées à une démonstration, séparées de l’évaluation. |
| `backend/ml_pipeline.py` | Lecture, audit, séparation, extraction HOG/HSV, mesures et inférence. |
| `backend/livrable2.py` | Les dix modèles, les expériences, la validation croisée et les analyses complémentaires. |
| `backend/requirements.txt` | Bibliothèques nécessaires à l’exécution. |
| `backend/__init__.py` | Déclare le package Python `backend`. |
| `backend/models/livrable2/` | Résultats, figures, traçabilité et modèle retenu. |
| `tests/` | Vérifications de séparation, sélection, rééchantillonnage, export et analyses par groupes. |
| `.gitignore` | Évite de versionner les caches et le modèle binaire généré. |

Les fonctions importées par le notebook restent dans les fichiers Python :
il faut conserver ces fichiers avec le notebook pour le réexécuter.

Dans `backend/models/livrable2/` :

| Fichier | Contenu |
|---|---|
| `run_status.json` | État de l’exécution : en cours ou terminée, avec version du protocole. |
| `validation_results.csv` | Les 60 configurations de présélection, scores, durées et diagnostics de convergence. |
| `baseline_results.csv` | Références majoritaire et aléatoire stratifiée, en validation et sur test. |
| `cross_validation_results.csv` | Chaque ajustement et chaque pli du réglage ciblé. |
| `cv_summary.csv` | Moyennes et écarts-types par configuration de validation croisée. |
| `cv_folds.csv` | Effectifs par classe dans chaque train et validation de pli. |
| `model_selection.json` | Présélection, paramètres retenus et décision figée avant le test. |
| `quality_audit.json`, `split_manifest.csv`, `split_summary.json` | Audit et séparation des observations. |
| `balancing_details.csv` | Effectifs du train et poids de classes. |
| `selected_features_validation.csv`, `selected_features_final.json` | Indices des variables retenues avant et après réentraînement. |
| `ablation_results.csv` | Comparaison des représentations HOG, HSV et de leur combinaison. |
| `learning_curve.csv` | Scores selon la taille réelle des sous-ensembles d’apprentissage groupés. |
| `learning_curve_class_counts.csv` | Effectifs par classe à chaque niveau, pour distinguer volume et composition. |
| `bootstrap_intervals.csv` | Intervalles exploratoires par groupes, avec fréquence des tirages à classes absentes. |
| `final_model_timings.csv` | Dimensions et coûts mesurés des dix finalistes. |
| `evaluation_metrics.json`, `test_predictions.csv`, `RESULTATS.md` | Mesures détaillées, prédictions et synthèse générée. |
| `figures/` | Distributions, exemples et matrices de confusion. |
| `selected_pipeline.joblib` | Modèle local, transformations et configuration pour l’inférence. |

Le modèle binaire est ignoré par Git et peut être régénéré par l’entraînement.

## Séparation et limites à expliquer

La validation initiale utilise des groupes source indivisibles, y compris les groupes
rapprochés par prudence à partir du pHash. Le rapprochement ne constitue pas une décision
de doublon. La règle choisit selon les effectifs parmi 1 024 partitions, sans utiliser
les scores des modèles.

| Classe | Entraînement interne | Validation initiale | Test |
|---|---:|---:|---:|
| bus | 680 | 120 | 200 |
| car | 680 | 120 | 200 |
| cyclist | 681 | 119 | 200 |
| motorcycle | 677 | 123 | 200 |
| pedestrian | 544 | 256 | 200 |
| **Total** | **3 262** | **738** | **1 000** |

La validation initiale représente 18,45 % du développement et 32 % de ses piétons.
Les piétons de développement proviennent de seulement **trois groupes : 531, 256 et
13 images**. La validation croisée groupée leur réserve donc successivement des
supports très différents. L’écart-type entre plis décrit cette variabilité ; il ne
doit pas être présenté comme une preuve de stabilité.

La présélection et le réglage utilisent le même développement. Cette validation croisée
sert à choisir un modèle et n’est pas une évaluation imbriquée indépendante.
L’ablation et la courbe d’apprentissage sont des diagnostics après sélection et ne
servent pas à remplacer le gagnant.

Le test avait déjà été évalué dans les versions antérieures du projet. Il ne sert pas
aux nouvelles décisions, mais une confirmation indépendante nécessite un nouveau test
externe. Ses piétons appartiennent à seulement **trois groupes : 144, 45 et 11 images**.
Certains groupes contiennent plusieurs classes : le bootstrap rééchantillonne des groupes
entiers et n’effectue pas un tirage séparé par classe. Les intervalles obtenus restent
exploratoires ; ils ne permettent pas d’affirmer une différence statistiquement significative
entre modèles.

Les temps du benchmark utilisent un cache partagé des prétraitements. Ils décrivent
cette exécution et ne constituent pas un classement universel de vitesse. Le temps
d’inférence est mesuré en batch, hors lecture de l’image et extraction HOG/HSV.
L’expérience concerne la classification d’objets déjà recadrés, avec des écarts de
provenance et des images apparentées ; elle ne valide pas un système de conduite autonome.

## Références méthodologiques

- [Modèles d’apprentissage supervisé — scikit-learn](https://scikit-learn.org/stable/supervised_learning.html)
- [Validation croisée — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Sélection des variables — scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)

## Préparation et contrôle

Le jeu a été préparé en amont avec des scripts : sélection, recadrage,
redimensionnement avec conservation des proportions et padding gris,
dédoublonnage et séparation des groupes source connus. Les données brutes et
les scripts historiques ne sont plus conservés. Aucun exemple synthétique ou
copie augmentée n’a été ajouté pour atteindre les effectifs.

Le notebook vérifie les images actuelles, pas l’historique de préparation.
Le dernier contrôle a trouvé zéro image illisible, zéro doublon exact,
trois paires similaires au sein de splits et une alerte d’exposition.
Deux paires de piétons semblent représenter des images proches de la même
personne. La classification les conserve dans leurs groupes source respectifs,
avec cette limite documentée ; aucune image n’est supprimée. L’image sombre
est également conservée. La revue des étiquettes n’est pas exhaustive.

`demo_scenes/` contient 20 scènes de rue séparées. Certaines partagent leurs
sources avec train : **ne pas les utiliser pour mesurer les performances**.

## Provenance à conserver

`sources.csv` décrit chaque image, sa source, son recadrage, sa licence connue,
son groupe source et son empreinte de pixels. `dataset/PROJECT_SPECIFICATION.md`
conserve le cahier des charges d’origine.

- [Traffic-Object, version 11](https://www.kaggle.com/datasets/tailength/traffic-object/versions/11) : 2 014 images. Licence annoncée CC BY-NC-SA 4.0 ; les sources sous-jacentes varient.
- [COCO 2017](https://cocodataset.org/#download) : 1 682 images. Licences et liens des photographies dans `sources.csv`.
- [Penn–Fudan](https://www.cis.upenn.edu/~jshi/ped_html/) : 400 piétons. Crédit : Liming Wang, Jianbo Shi, Gang Song et I-fan Shen, *Object Detection Combining Recognition and Segmentation*, ACCV 2007. Consulter les conditions de la source.
- Jeu initial fourni par l’utilisateur : 904 images ; licences non fournies.

La classe `cyclist` comprend les vélos seuls ou avec cycliste, conformément au
guide ; `motorcycle` comprend motos et scooters. Certaines images de deux-roues
sont des photos de produit. Les mesures de classification sont présentées
dans le notebook du livrable 2 et dans `backend/models/livrable2/RESULTATS.md`.

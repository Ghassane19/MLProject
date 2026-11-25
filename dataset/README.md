# Jeu d’images routières

5 000 images RGB PNG de 128 × 128 pixels : **800 train et 200 test par classe**.
Classes : `bus`, `car`, `cyclist`, `motorcycle`, `pedestrian`.

Le notebook `../notebooks/Nettoyage_des_donnees.ipynb` contient le code, les explications
et les résultats du contrôle de qualité. Il fonctionne sans script Python
externe et sans dossier de rapports. Les images originales restent inchangées.

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
personne. La classification les conserve dans leurs groupes source avec une
limite documentée ; aucune image n’est supprimée. La revue des étiquettes
n’est pas exhaustive.

`demo_scenes/` contient 20 scènes de rue séparées. Certaines partagent leurs
sources avec train : **ne pas les utiliser pour mesurer les performances**.

## Provenance à conserver

`sources.csv` décrit chaque image, sa source, son recadrage, sa licence connue,
son groupe source et son empreinte de pixels. `PROJECT_SPECIFICATION.md`
conserve le cahier des charges d’origine.

- [Traffic-Object, version 11](https://www.kaggle.com/datasets/tailength/traffic-object/versions/11) : 2 014 images. Licence annoncée CC BY-NC-SA 4.0 ; les sources sous-jacentes varient.
- [COCO 2017](https://cocodataset.org/#download) : 1 682 images. Licences et liens des photographies dans `sources.csv`.
- [Penn–Fudan](https://www.cis.upenn.edu/~jshi/ped_html/) : 400 piétons. Crédit : Liming Wang, Jianbo Shi, Gang Song et I-fan Shen, *Object Detection Combining Recognition and Segmentation*, ACCV 2007. Consulter les conditions de la source.
- Jeu initial fourni par l’utilisateur : 904 images ; licences non fournies.

La classe `cyclist` comprend les vélos seuls ou avec cycliste, conformément au
guide ; `motorcycle` comprend motos et scooters. Certaines images de deux-roues
sont des photos de produit. La classification et l’évaluation sont désormais
présentées dans `../notebooks/Livrable_2_Preparation_Selection_Entrainement.ipynb` et
`../backend/models/livrable2/RESULTATS.md`. Les paires similaires sont conservées avec
protection par groupes et limites documentées ; aucune image n’est supprimée.

## Utilisation dans le livrable 2 étendu

Le protocole `expanded-v2` compare dix modèles sur les mêmes descripteurs HOG/HSV,
avec trois sélections de variables et deux stratégies de gestion du déséquilibre,
soit 60 configurations initiales. Les trois modèles présélectionnés sont réglés
sur une validation croisée à trois plis par groupes. La sélection du modèle final
utilise le développement uniquement ; le test existant sert ensuite au bilan des
dix finalistes.

Les transformations, les sélections et les éventuels rééchantillonnages sont
ajustés sur le train de chaque expérience. La validation et le test ne sont
jamais rééquilibrés. La courbe d’apprentissage conserve également les groupes
entiers dans des sous-ensembles emboîtés du train.

La correction de k-NN, GaussianNB, HistGradientBoosting et MLP duplique uniquement
des lignes d’entraînement en mémoire. Pour HistGradientBoosting, ce choix évite
le surcoût observé du calcul pondéré des histogrammes ; les autres modèles
compatibles utilisent des poids de classes. Aucun fichier image n’est dupliqué.

Les piétons ne disposent que de trois groupes de développement (531, 256 et
13 images) et de trois groupes test (144, 45 et 11 images). Les supports de
validation croisée sont donc très différents. Les intervalles bootstrap
rééchantillonnent les groupes test entiers, y compris ceux qui contiennent
plusieurs classes, et restent exploratoires. Les tirages sans certaines classes
sont conservés avec une définition fixe des cinq classes dans les scores.

Le test a déjà été utilisé dans les versions précédentes du projet : un nouveau
jeu externe serait nécessaire pour une confirmation indépendante. Les détails
du protocole, les fichiers produits et les limites sont décrits dans le
[README du projet](../README.md) et dans le notebook du livrable 2.

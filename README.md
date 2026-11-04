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
personne. Elles restent à traiter ; aucune image n’a été supprimée dans cet
audit. La revue des étiquettes n’est pas exhaustive.

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
sont des photos de produit. La classification et l’évaluation des modèles
restent à réaliser.

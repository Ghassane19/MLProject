# Livrable 2 — Comparaison étendue

**10 modèles, 60 configurations de présélection, 30 ajustements en validation croisée.**

Modèle retenu parmi les trois familles présélectionnées : **SVM**.
Sélection : Toutes ; correction : Sans_correction ; paramètres : {"C": 10}.
F1 macro CV moyen : **0.8345** (écart-type entre plis : 0.0276).
Accuracy test : **81.90%** ; F1 macro test : **0.8178**.

## Dix finalistes

| Modèle | Accuracy test | F1 macro test |
|---|---:|---:|
| SVM | 81.90% | 0.8178 |
| SGD | 75.70% | 0.7535 |
| LogisticRegression | 76.90% | 0.7679 |
| RandomForest | 75.10% | 0.7478 |
| ExtraTrees | 73.50% | 0.7312 |
| kNN | 74.70% | 0.7445 |
| DecisionTree | 55.40% | 0.5528 |
| GaussianNB | 71.50% | 0.7079 |
| HistGradientBoosting | 79.40% | 0.7918 |
| MLP | 80.50% | 0.8037 |

## Analyses complémentaires

- Références majoritaire et aléatoire stratifiée.
- ANOVA et importance Random Forest ; variables retenues exportées.
- Déséquilibre traité sur les ajustements train seulement.
- Réglage ciblé en validation croisée groupée ; effectifs de chaque pli publiés.
- Ablation HOG / HSV / combinaison, à paramètres fixés et sans nouvelle sélection du gagnant.
- Courbe d'apprentissage sur des sous-ensembles emboîtés à groupes indivisibles.
- Temps d'apprentissage, dimensions, temps d'inférence en batch et diagnostics de convergence.
- Précision/rappel/F1 par classe, erreurs visuelles et résultats par source.
- Intervalles bootstrap par groupes, exploratoires.

## Limites à défendre devant le jury

Le test a déjà été évalué dans les versions antérieures du projet. Les nouvelles décisions utilisent uniquement le développement. Ces résultats restent exploratoires et demandent un nouveau test externe.

La validation croisée règle les paramètres de trois familles présélectionnées sur une validation issue du même développement. Elle n'est pas imbriquée et ne fournit pas une estimation indépendante de généralisation.

Les piétons de développement n'ont que trois groupes (531, 256 et 13 images). Les plis ne sont donc pas équivalents en support. L'écart-type CV décrit cette variabilité.

Intervalles percentile 95 % sur 1 000 tirages de groupes entiers, avec cinq classes fixes ; les tirages sans certaines classes sont inclus. Seulement trois groupes de piétons : interprétation exploratoire, sans conclusion de significativité statistique.

Les temps de benchmark bénéficient d'un cache partagé des prétraitements ; ils décrivent cette exécution, pas un classement universel de vitesse. L'inférence est mesurée en batch, hors lecture d'image et extraction HOG/HSV.

- Les étiquettes et les relations entre sources ne sont pas certifiées exhaustivement.
- Le pHash signale des ressemblances, sans prouver que deux objets sont identiques.
- Les groupes source sont indivisibles : 15 % par classe n'est pas toujours réalisable.
- Quelques grands groupes de piétons limitent la représentativité de la validation ; l'écart de domaine entre sources peut influencer les scores.
- Les images apparentées restent dans leur groupe ; les observations du test ne sont donc pas toutes indépendantes. Les scores ne prouvent pas une robustesse en circulation réelle.
- Les probabilités éventuellement fournies ne sont pas calibrées pour un usage réel.
- Les photos de produit, les fonds et les recadrages peuvent créer des raccourcis visuels.
- Les explications des confusions sont des hypothèses à vérifier sur les images.

## Suite du projet

- Acquérir de nouveaux groupes et un test externe tenu à l'écart.
- Étendre le réglage en validation imbriquée lorsque les groupes indépendants le permettent.
- Vérifier les annotations et analyser les écarts de domaine et les erreurs difficiles.
- Comparer un futur transfert d'apprentissage dans une expérience séparée et équitable.

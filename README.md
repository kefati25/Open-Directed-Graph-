## Description
Ce projet implémente une structure de données pour représenter et manipuler des graphes orientés avec des nœuds d'entrée et de sortie. Il inclut également une extension pour les circuits booléens avec des fonctionnalités d'évaluation et de transformation.

___


## Fonctionnalités principales
### Classes principales
node: Représente un nœud dans le graphe avec des propriétés comme:

Identifiant unique, étiquette, parents et enfants avec multiplicité, méthodes pour manipuler les connexions

open_digraph: Représente un graphe orienté avec:

Nœuds d'entrée et de sortie, dictionnaire de nœuds, fonctions pour manipuler la structure du graphe, validation de la forme correcte du graphe, génération de matrices d'adjacence, import/export vers des fichiers .dot, opérations sur les graphes (composition, parallèle, etc.), algorithmes de graphes (Dijkstra, composantes connexes, etc.)

bool_circ: Extension pour les circuits booléens avec:

Validation des circuits booléens, évaluation des portes logiques, transformations spécifiques aux circuits booléen

### Fonctionnalités avancées
Génération de graphes aléatoires avec différentes contraintes

Manipulation de matrices d'adjacence

Algorithmes de parcours de graphes

Opérations sur les circuits booléens (portes ET, OU, NON, XOR)

Optimisations de circuits booléens

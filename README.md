## Description
Ce projet implémente une structure de données pour représenter et manipuler des graphes orientés avec des nœuds d'entrée et de sortie. Il inclut également une extension pour les circuits booléens avec des fonctionnalités d'évaluation et de transformation.

## Fonctionnalités principales
### Classes principales
node: Représente un nœud dans le graphe avec des propriétés comme:
Identifiant unique
Étiquette
Parents et enfants avec multiplicité
Méthodes pour manipuler les connexions

open_digraph: Représente un graphe orienté avec:
Nœuds d'entrée et de sortie
Dictionnaire de nœuds
Fonctions pour manipuler la structure du graphe
Validation de la forme correcte du graphe
Génération de matrices d'adjacence
Import/export vers des fichiers .dot
Opérations sur les graphes (composition, parallèle, etc.)
Algorithmes de graphes (Dijkstra, composantes connexes, etc.)

bool_circ: Extension pour les circuits booléens avec:
Validation des circuits booléens
Évaluation des portes logiques
Transformations spécifiques aux circuits booléen

### Fonctionnalités avancées
Génération de graphes aléatoires avec différentes contraintes
Manipulation de matrices d'adjacence
Algorithmes de parcours de graphes
Opérations sur les circuits booléens (portes ET, OU, NON, XOR)
Optimisations de circuits booléens

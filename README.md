# Système de Détection d'Intrusion Basé sur l'Apprentissage Automatique

Ce projet implémente un système de détection d'intrusion (IDS) utilisant plusieurs algorithmes d'apprentissage automatique sur le jeu de données KDD Cup 1990. L'objectif principal est de détecter des intrusions dans un réseau en classifiant les connexions réseau comme normales ou malveillantes.

## Table des Matières
- [Contexte](#contexte)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Modèles Utilisés](#modèles-utilisés)
- [Conclusion](#conclusion)
- [Auteurs](#auteurs)

## Contexte
Le projet s'appuie sur le jeu de données KDD Cup 1990, qui est largement utilisé pour le développement et l'évaluation des systèmes de détection d'intrusion. Les données comprennent diverses caractéristiques des connexions réseau, qui sont utilisées pour prédire si la connexion est une intrusion ou non.

## Installation
1. Clonez le dépôt :
   
   ```bash
   git clone https://github.com/votre-utilisateur/nom-du-repo.git
   cd nom-du-repo
3. Installez les dépendances requises :
   
   ```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib

## Utilisation
1. Assurez-vous d'avoir le fichier de données kddcup.data_10_percent dans le répertoire de travail.
2. Exécutez le script :
   ```bash
python IDS_PROJECT.py

Ce script chargera les données, les prétraitera, appliquera l'oversampling SMOTE pour équilibrer les classes, et entraînera plusieurs modèles d'apprentissage automatique.

## Résultats
Le script affiche :

Un aperçu des données.
- La distribution des classes avant rééchantillonnage.
- Des rapports de classification pour chaque modèle.
- Des matrices de confusion pour visualiser les performances des modèles.
- Des courbes ROC pour évaluer les performances des modèles.

## Modèles Utilisés
Le projet teste les modèles suivants :

- Support Vector Machine (SVM)
- Forêt Aléatoire
- Réseau de Neurones Multi-Couches (MLP)
- Régression Logistique
- Arbre de Décision

Les modèles sont entraînés et évalués sur un ensemble de données d'entraînement et de test.

## Conclusion
Ce projet démontre comment utiliser l'apprentissage automatique pour détecter des intrusions dans un réseau en utilisant des techniques de prétraitement des données et d'équilibrage des classes. Les résultats peuvent être améliorés en explorant d'autres modèles ou en ajustant les hyperparamètres.

## Auteurs
Sun PIAHA

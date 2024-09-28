# Projet : Système de Détection d'Intrusion (IDS) Basé sur l'Apprentissage Automatique

# Description
Ce projet implémente un système de détection d'intrusion (IDS) utilisant plusieurs algorithmes de machine learning pour détecter des comportements anormaux dans les réseaux. Les modèles entraînés incluent SVM, Forêt Aléatoire, Réseau de Neurones, Régression Logistique et Arbre de Décision. Une interface utilisateur est développée avec Flask pour permettre le téléchargement et l'analyse d'ensembles de données personnalisés.

# Fonctionnalités
Précision, Rappel, F1-score : Évaluation de la performance des modèles.
Rééchantillonnage : Utilisation de SMOTE pour équilibrer les classes.
Visualisation : Graphiques avec Matplotlib et Seaborn.
Interface Flask : Téléchargement d'ensembles de données personnalisés.
Support de Modèles : SVM, Forêt Aléatoire, Réseau de Neurones, Régression Logistique, Arbre de Décision.
Réduction de dimensionnalité : PCA pour visualisation des données.

# Prérequis
Python 3.x
Packages : pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, imblearn, flask

# Installation
Clonez le dépôt GitHub :

git clone https://github.com/votre_utilisateur/ids_project.git
cd ids_project
Créez et activez un environnement virtuel :
bash
Copier le code
python -m venv myenv
source myenv/bin/activate  # Sous Windows : myenv\Scripts\activate
Installez les dépendances :
bash
Copier le code
pip install -r requirements.txt

# Utilisation
Exécution du script principal :

bash
Copier le code
python script.py
Lancer l'interface Flask :

bash
Copier le code
export FLASK_APP=app.py
flask run
Accédez à l'interface via http://127.0.0.1:5000.

# Fonctionnalités Clés
Rééchantillonnage avec SMOTE : Gère les classes déséquilibrées pour améliorer les performances de détection.
Visualisation PCA : Permet de comprendre la distribution des données.
Évaluation des Modèles : Fournit des métriques de performance détaillées pour chaque modèle.
Interface Utilisateur Conviviale : Les utilisateurs peuvent facilement télécharger et analyser leurs propres ensembles de données.

# Exemple de Sortie
yaml
Copier le code
Distribution des classes avant rééchantillonnage :
normal.             7787
smurf.              2207
buffer_overflow.       2
neptune.               2
loadmodule.            1
perl.                  1
Name: count, dtype: int64

Aperçu des données :
   duree type_protocole  ... taux_erreur_srv_hote_dst etiquette
0      0            tcp  ...                      0.0   normal.
1      0            tcp  ...                      0.0   normal.
2      0            tcp  ...                      0.0   normal.
3      0            tcp  ...                      0.0   normal.
4      0            tcp  ...                      0.0   normal.

--- SVM ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2351
           1       1.00      1.00      1.00      2322

    accuracy                           1.00      4673
   macro avg       1.00      1.00      1.00      4673
weighted avg       1.00      1.00      1.00      4673

# Auteur
PIAHA SUN

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

# Charger le fichier de données
donnees = pd.read_csv('kddcup.data_10_percent', header=None, nrows=10000)

# Définir les noms des colonnes
noms_colonnes = [
    "duree", "type_protocole", "service", "drapeau", "octets_src", "octets_dst",
    "land", "fragment_errone", "urgent", "hot", "nombre_echecs_connexions", "connecte",
    "nombre_compromis", "root_shell", "tentative_su", "nombre_root", "nombre_creations_fichiers",
    "nombre_shells", "nombre_fichiers_acces", "nombre_commandes_sortantes", "login_hote",
    "login_invite", "compte", "srv_compte", "taux_erreur_serr", "taux_erreur_srv_serr", 
    "taux_erreur", "taux_erreur_srv", "taux_meme_srv", "taux_diff_srv",
    "taux_srv_diff_hote", "compte_hote_dst", "srv_compte_hote_dst", 
    "taux_meme_srv_hote_dst", "taux_diff_srv_hote_dst",
    "taux_meme_src_port_hote_dst", "taux_srv_diff_hote_dst",
    "taux_erreur_serr_hote_dst", "taux_erreur_srv_serr_hote_dst", "taux_erreur_hote_dst",
    "taux_erreur_srv_hote_dst", "etiquette"
]

donnees.columns = noms_colonnes

# Encoder les variables catégoriques
donnees_encodees = pd.get_dummies(donnees, columns=["type_protocole", "service", "drapeau"])

# Séparer les caractéristiques et les étiquettes
X = donnees_encodees.drop(columns=["etiquette"])
y = donnees_encodees["etiquette"]

# Vérifiez la distribution des classes
print("Distribution des classes avant rééchantillonnage :")
print(y.value_counts())

# Normaliser les caractéristiques
normaliseur = StandardScaler()
X_normalise = normaliseur.fit_transform(X)

# Gérer l'imbalance des données
# Retirer les classes avec très peu d'échantillons
classes_a_conserver = y.value_counts()[y.value_counts() >= 5].index
donnees_filtrees = donnees_encodees[donnees_encodees['etiquette'].isin(classes_a_conserver)]
X = donnees_filtrees.drop(columns=["etiquette"])
y = donnees_filtrees["etiquette"]

# Appliquer SMOTE uniquement si la classe minoritaire a au moins 2 échantillons
if y.value_counts().min() >= 2:
    smote = SMOTE(k_neighbors=1)
    X_resample, y_resample = smote.fit_resample(normaliseur.transform(X), y)
else:
    raise ValueError("Il n'y a pas assez d'échantillons dans la classe minoritaire pour appliquer SMOTE.")

# Encoder les étiquettes en nombres pour la visualisation
label_encoder = LabelEncoder()
y_resample_encoded = label_encoder.fit_transform(y_resample)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample_encoded, test_size=0.3, random_state=42)

# Réduction de dimensionnalité avec PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_resample)

# Visualisation des données après PCA
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_resample_encoded, cmap='viridis')
plt.title("Visualisation des données après PCA")
plt.xlabel("Composante PCA 1")
plt.ylabel("Composante PCA 2")
plt.colorbar(ticks=range(len(label_encoder.classes_)), label='Classes')
plt.clim(-0.5, len(label_encoder.classes_) - 0.5)
plt.show()

# Créer une liste de modèles à tester
modeles = {
    "SVM": SVC(probability=True),
    "Forêt Aléatoire": RandomForestClassifier(),
    "Réseau de Neurones": MLPClassifier(max_iter=300),
    "Régression Logistique": LogisticRegression(max_iter=300),
    "Arbre de Décision": DecisionTreeClassifier()
}

# Affichage d'un aperçu des données
print("Aperçu des données :")
print(donnees.head())

# Itérer sur chaque modèle, l'entraîner et l'évaluer
for nom_modele, modele in modeles.items():
    print(f"\n--- {nom_modele} ---")
    
    # Entraîner le modèle
    modele.fit(X_train, y_train)
    
    # Faire des prédictions sur l'ensemble de test
    y_pred = modele.predict(X_test)
    
    # Afficher le rapport de classification
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    matrice_confusion = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrice_confusion, annot=True, fmt="d")
    plt.title(f'Matrice de Confusion - {nom_modele}')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.show()
    
    # Courbe ROC
    y_proba = modele.predict_proba(X_test)[:, 1] if hasattr(modele, "predict_proba") else modele.decision_function(X_test)

    # Spécifiez le label positif pour le ROC
    pos_label = 1 if len(label_encoder.classes_) > 1 else 0
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=pos_label)
    score_auc = auc(fpr, tpr)
    
    # Tracer la courbe ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (aire = {score_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Caractéristique de Fonctionnement du Récepteur - {nom_modele}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Sauvegarder le modèle
    joblib.dump(modele, f'{nom_modele}_model.pkl')

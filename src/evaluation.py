from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

def train_and_evaluate(pipeline, X_train, y_train, X_val, y_val):
    """
    Entraîne une pipeline et affiche les classification reports 
    pour les ensembles de Train et de Validation.
    """
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)

    print("--------- Métriques pour TRAIN ---------")
    print(classification_report(y_train, y_train_pred))

    y_val_pred = pipeline.predict(X_val)
    
    print("--------- Métriques pour VALIDATION ---------")
    print(classification_report(y_val, y_val_pred))

    return pipeline


def plot_confusion_matrix(model, X, y_true, title="Matrice de confusion"):
    """
    Génère les prédictions et affiche la matrice de confusion sous forme de heatmap.
    
    Parameters:
    -----------
    model : pipeline ou estimateur entraîné
        Le modèle à évaluer
    X : DataFrame ou array
        Les variables explicatives (Features)
    y_true : Series ou array
        La variable cible réelle (Target)
    title : str
        Titre du graphique
    """
    # Génération automatique des prédictions
    y_pred = model.predict(X)
    
    # Calcul de la matrice
    cm = confusion_matrix(y_true, y_pred)

    # Affichage graphique
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=["Solvable (0)", "Défaut (1)"],
        yticklabels=["Solvable (0)", "Défaut (1)"],
        cbar=True,
        annot_kws={'size': 14, 'weight': 'bold'}
    )

    # Labeling propre
    plt.xlabel("Prédit", fontsize=12, labelpad=10)
    plt.ylabel("Réel", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, weight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()

    return cm


def plot_models_pr_curves(models_dict, X_val, y_val, title="Comparaison des Courbes Précision-Rappel (PRC)"):
    """
    Calcule les scores PR-AUC et trace les courbes Précision-Rappel 
    de plusieurs modèles sur un même graphique.

    Parameters:
    -----------
    models_dict : dict
        Dictionnaire contenant les pipelines { "Nom du Modèle": pipeline_instanciée }
    X_val : DataFrame ou array
        Features du jeu de validation
    y_val : Series ou array
        Cible du jeu de validation
    title : str
        Titre du graphique
    """
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    print("--- Scores PR-AUC (Average Precision) ---")
    
    # Boucle sur chaque modèle du dictionnaire pour calculer le score et tracer la courbe
    for name, pipeline in models_dict.items():
        # Récupération des probabilités de la classe 1
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        score_pr_auc = average_precision_score(y_val, y_proba)
        
        print(f"{name} : {score_pr_auc:.3f}")
        
        # Ajout de la courbe sur le graphique commun
        #PrecisionRecallDisplay.from_estimator(pipeline, X_val, y_val, ax=ax, name=f"{name} (PR-AUC = {score_pr_auc:.3f})")

    # Configuration et habillage du graphique
    plt.title(title, fontsize=14)
    plt.xlabel("Rappel (Recall)")
    plt.ylabel("Précision")
    #plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
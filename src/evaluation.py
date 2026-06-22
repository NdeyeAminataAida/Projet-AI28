import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import cross_val_score
import optuna


def _get_positive_scores(model, X):
    '''
    Renvoie un score continu pour la classe "défaut" (1), quel que soit le modèle.

    La plupart des modèles exposent predict_proba ; la SVM (probability=False)
    n'expose que decision_function. On uniformise ici pour que le calcul du PR-AUC
    et du ROC-AUC fonctionne sur TOUS les modèles du panel sans cas particulier.
    '''
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_and_evaluate(pipeline, X_train, y_train, X_val, y_val,threshold=0.5):
    """
    Entraîne une pipeline et affiche les classification reports 
    pour les ensembles de Train et de Validation en appliquant un seuil personnalisé.
    """
    pipeline.fit(X_train, y_train)

    # Évaluation Train avec seuil personnalisé
    y_train_proba = _get_positive_scores(pipeline, X_train)
    y_train_pred = (y_train_proba >= threshold).astype(int)

    print(f"--------- Métriques pour TRAIN(Seuil : {threshold}) ---------")
    print(classification_report(y_train, y_train_pred))

   # Évaluation Validation avec seuil personnalisé
    y_val_proba = _get_positive_scores(pipeline, X_val)
    y_val_pred = (y_val_proba >= threshold).astype(int)
    
    print(f"--------- Métriques pour VALIDATION(Seuil : {threshold}) ---------")
    print(classification_report(y_val, y_val_pred))

    return pipeline


def plot_confusion_matrix(model, X, y_true, title="Matrice de confusion",threshold=0.5):
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
    ## Application rigoureuse du seuil personnalisé sur les probabilités calculées
    y_proba = _get_positive_scores(model, X)
    y_pred = (y_proba >= threshold).astype(int)
    
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


def run_model_comparison(model_configs, X_train, y_train, X_val, y_val,
                         scoring="average_precision", cv=5, n_trials=25, random_state=42,threshold=0.5):
    '''
    Orchestration unique de toute la modélisation, pilotée par Optuna.

    Itère sur le registre de modèles (cf. modeling.get_model_configurations) et,
    pour chacun : lance une étude Optuna (hyper-optimisation bayésienne via
    l'échantillonneur TPE) qui maximise le score de validation croisée, puis
    réentraîne la meilleure pipeline sur tout X_train et affiche le
    classification_report sur le jeu de validation. Évite ainsi de répéter le même
    code d'optimisation pour chaque modèle.

    Fonction objectif : pour chaque essai, Optuna tire des hyper-paramètres
    (config["suggest"]), les applique à une pipeline neuve (config["build"]) et
    renvoie la moyenne du score en validation croisée (cross_val_score). On maximise
    par défaut le PR-AUC (scoring="average_precision") plutôt que l'accuracy : c'est
    la métrique adaptée au déséquilibre (~22 % de défauts), et celle utilisée pour la
    comparaison finale, d'où la cohérence.

    n_trials règle le budget d'essais Optuna (un modèle peut le surcharger via la clé
    "n_trials" du registre, ex. la SVM coûteuse). On fixe la graine du sampler pour
    rendre l'optimisation reproductible.

    Renvoie un dict {nom du modèle : meilleure pipeline entraînée}, directement
    réutilisable par summarize_models() et plot_models_pr_curves().
    '''
    # On coupe les logs "INFO" d'Optuna (un message par essai) pour garder une sortie lisible
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fitted_models = {}

    for name, config in model_configs.items():
        print(f"\n===== {name} =====")

        # Fonction objectif : score CV moyen pour un jeu d'hyper-paramètres tiré par Optuna.
        # _config en argument par défaut pour figer la bonne config dans la closure.
        def objective(trial, _config=config):
            params = _config["suggest"](trial)
            model = _config["build"]()
            model.set_params(**params)
            return cross_val_score(
                model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1
            ).mean()

        trials = config.get("n_trials", n_trials)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        print(f"Optimisation Optuna en cours ({trials} essais)...")
        study.optimize(objective, n_trials=trials)

        print("Meilleurs hyper-paramètres :", study.best_params)
        print(f"Meilleur score CV ({scoring}) : {study.best_value:.3f}")

        # Réentraînement de la meilleure pipeline sur tout X_train (Optuna ne le fait pas seul).
        # study.best_params est clé-compatible avec set_params (cf. get_model_configurations).
        best_model = config["build"]()
        best_model.set_params(**study.best_params)
        best_model.fit(X_train, y_train)

        print(f"--------- Métriques VALIDATION (Seuil : {threshold}) ---------")
        y_val_proba = _get_positive_scores(best_model, X_val)
        y_val_pred = (y_val_proba >= threshold).astype(int)
        print(classification_report(y_val, y_val_pred))

        fitted_models[name] = best_model

    return fitted_models


class _KeyRecorder:
    '''
    Faux "trial" Optuna : on l'appelle à la place d'un vrai trial pour récupérer
    les NOMS des hyper-paramètres qu'un modèle optimise (sans lancer d'optimisation).
    Les valeurs renvoyées sont factices ; seules les clés enregistrées nous importent.
    '''
    def __init__(self):
        self.keys = []

    def suggest_float(self, name, *args, **kwargs):
        self.keys.append(name)
        return 1.0

    def suggest_int(self, name, *args, **kwargs):
        self.keys.append(name)
        return 1

    def suggest_categorical(self, name, choices, **kwargs):
        self.keys.append(name)
        return choices[0]


def summarize_hyperparameters(model_configs, fitted_models):
    '''
    Tableau récapitulatif des hyper-paramètres retenus par Optuna pour chaque modèle.

    Pour chaque modèle, on retrouve les noms des hyper-paramètres optimisés (en
    "rejouant" sa fonction suggest sur un faux trial), puis on lit leur valeur
    effective dans la pipeline entraînée (fitted_models). On obtient un tableau au
    format long : une ligne par couple (modèle, hyper-paramètre), pratique à afficher
    dans le rapport à côté du tableau des PR-AUC.
    '''
    class _KR:
        def __init__(self): self.keys = []
        def suggest_float(self, name, *a, **kw): self.keys.append(name); return 1.0
        def suggest_int(self, name, *a, **kw): self.keys.append(name); return 1
        def suggest_categorical(self, name, choices, **kw): self.keys.append(name); return choices[0]

    rows = []
    for name, config in model_configs.items():
        recorder = _KR()
        config["suggest"](recorder)  # remplit recorder.keys avec les noms optimisés
        params = fitted_models[name].get_params()
        for key in recorder.keys:
            value = params[key]
            if isinstance(value, float):
                value = round(value, 4)
            rows.append({
                "Modèle": name,
                "Hyper-paramètre": key.split("__")[-1],  # on retire le préfixe de l'étape
                "Valeur retenue": value,
            })
    return pd.DataFrame(rows)


def summarize_models(models_dict, X_val, y_val,threshold=0.5):
    '''
    Construit l'UNIQUE tableau récapitulatif comparant tous les modèles optimisés
    sur le jeu de validation (au lieu de recalculer les métriques modèle par modèle).

    Les métriques précision/rappel/F1 sont données pour la classe "défaut" (1),
    la classe d'intérêt. Le tableau est trié par PR-AUC décroissant, ce qui place
    en tête le modèle recommandé pour ce problème déséquilibré.
    '''
    rows = []
    for name, model in models_dict.items():
        y_scores = _get_positive_scores(model, X_val)
        y_pred = (y_scores >= threshold).astype(int)
        rows.append({
            "Modèle": name,
            "Précision (défaut)": precision_score(y_val, y_pred),
            "Rappel (défaut)": recall_score(y_val, y_pred),
            "F1 (défaut)": f1_score(y_val, y_pred),
            "PR-AUC": average_precision_score(y_val, y_scores),
            "ROC-AUC": roc_auc_score(y_val, y_scores),
        })

    tableau = (
        pd.DataFrame(rows)
        .sort_values("PR-AUC", ascending=False)
        .reset_index(drop=True)
    )
    return tableau


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
        # Score continu de la classe 1 (predict_proba, ou decision_function pour la SVM)
        y_proba = _get_positive_scores(pipeline, X_val)
        score_pr_auc = average_precision_score(y_val, y_proba)
        
        print(f"{name} : {score_pr_auc:.3f}")
        
        # Ajout de la courbe sur le graphique commun
        PrecisionRecallDisplay.from_estimator(pipeline, X_val, y_val, ax=ax, name=f"{name} (PR-AUC = {score_pr_auc:.3f})")

    # Configuration et habillage du graphique
    plt.title(title, fontsize=14)
    plt.xlabel("Rappel (Recall)")
    plt.ylabel("Précision")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize=10, 
        prop={'size': 10, 'weight': 'normal'})
    plt.grid(True, linestyle="--", alpha=0.6)
   # Pour forcer le graphique à garder une forme carrée et laisser 35% d'espace libre à droite pour la légende
    plt.gcf().set_size_inches(9, 6) # Donne une taille globale propre au graphique
    plt.subplots_adjust(right=0.65, left=0.1, top=0.9, bottom=0.1)
    plt.show()
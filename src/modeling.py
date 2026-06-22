from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_tree_preprocessor, build_logistic_preprocessor,build_robust_preprocessor

'''
On définit ici les fonctions qui construisent les modèles.
Chaque fonction renvoie une pipeline complète (pré-traitement + modèle)
prête à être passée à train_and_evaluate() de evaluation.py.
'''

    

def build_logistic_model(C=1.0, class_weight=None):
    '''
    Construit une pipeline complète de Régression Logistique (pré-traitement + modèle).

    Le modèle cherche à tracer une frontière linéaire entre les classes. Il nécessite
    une mise à l'échelle (StandardScaler) des montants et de l'âge pour converger correctement,
    ainsi qu'un encodage disjonctif de l'historique et des catégories.

    C représente le param de régularisation (1/lambda dans le cours)
    '''
    preprocessor = build_logistic_preprocessor()

    # Configuration du classificateur linéaire
    classifier = LogisticRegression(
        C=C, class_weight=class_weight, max_iter=1000, random_state=42
    )

    # Assemblage de la pipeline
    pipeline = make_pipeline(preprocessor, classifier)
    return pipeline

def build_decision_tree_model(max_depth=None, class_weight=None):
    """
    Construit une pipeline complète d'Arbre de Décision simple.
    
    Les arbres n'ont pas besoin de normalisation des montants, 
    on utilise donc le préprocesseur minimal (build_tree_preprocessor).
    """
    preprocessor = build_tree_preprocessor()
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42
    )
    return make_pipeline(preprocessor, classifier)

def build_random_forest_model(n_estimators=100, max_depth=None, class_weight=None):
    """
    Construit une pipeline complète de Random Forest (pré-traitement + modèle).
    
    Modèle d'ensemble par Bagging. Idéal pour réduire la variance d'arbres simples.
    """
    preprocessor = build_tree_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1  # Utilise tous les cœurs du processeur pour accélérer l'entraînement
    )
    return make_pipeline(preprocessor, classifier)


def build_adaboost_model(n_estimators=100, learning_rate=1.0, max_depth=1, class_weight=None):
    '''
    Construit une pipeline AdaBoost (méthode d'ensemble séquentielle / boosting).

    Principe (cf. cours chap.9) : on entraîne K modèles faibles l'un après l'autre,
    chacun se concentrant sur les observations mal classées par son prédécesseur,
    puis on agrège leurs prédictions par un vote pondéré.

    class_weight="balanced" pondère les classes à l'inverse de leur fréquence :
    utile ici car seulement ~22% des clients font défaut (classe minoritaire).
    '''
    # On réutilise le pré-processeur "arbres" : les arbres n'ont pas besoin de
    # normalisation, on encode juste les catégorielles (SEX, EDUCATION, MARRIAGE)
    preprocessor = build_tree_preprocessor()

    # Le prédicteur faible (weak learner) : un arbre très court (stump si max_depth=1),
    # qui fait à peine mieux que le hasard mais devient fort une fois agrégé
    weak_learner = DecisionTreeClassifier(
        max_depth=max_depth, class_weight=class_weight, random_state=42
    )

    model = make_pipeline(
        preprocessor,
        AdaBoostClassifier(
            estimator=weak_learner,
            n_estimators=n_estimators,      # K : nombre de prédicteurs de la foule
            learning_rate=learning_rate,    # eta : dose la contribution de chaque modèle
            random_state=42
        )
    )
    return model


def build_gradient_boost_model(n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Construit une pipeline complète de Gradient Boosting.
    
    Modèle d'ensemble séquentiel puissant. Il construit des arbres successifs
    pour corriger les erreurs des précédents en utilisant la descente de gradient.
    """
    preprocessor = build_tree_preprocessor()
    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    return make_pipeline(preprocessor, classifier)

def build_knn_model(n_neighbors=5, weights="uniform"):
    '''
    Construit une pipeline complète de K plus proches voisins (cf. notes cours 9).

    Le KNN classe un client selon le vote majoritaire de ses k voisins les plus
    proches. Comme il repose sur des distances, il EXIGE une mise à l'échelle :
    on utilise donc le pré-processeur "à distance" (build_logistic_preprocessor).

    Remarque : KNeighborsClassifier ne propose pas de class_weight ; le déséquilibre
    se gère plutôt via weights="distance" (les voisins proches pèsent davantage).
    '''
    preprocessor = build_robust_preprocessor()
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1  # parallélise la recherche des voisins
    )
    return make_pipeline(preprocessor, classifier)


def build_svm_model(C=1.0, gamma="scale", kernel="rbf", class_weight=None):
    '''
    Construit une pipeline complète de Machine à Vecteurs de Support (cf. slides7).

    La SVM cherche l'hyperplan de marge maximale séparant les classes ; le noyau
    RBF autorise une frontière non linéaire. Comme la régression logistique, elle
    nécessite une mise à l'échelle des variables (build_logistic_preprocessor).

    C règle le compromis marge / erreurs (1/lambda du cours), gamma la portée du noyau.
    class_weight="balanced" pénalise davantage les erreurs sur la classe minoritaire
    (~22 % de défauts). On laisse probability=False : pour les courbes PR on s'appuie
    sur decision_function, ce qui évite la validation croisée interne (très coûteuse ici).
    '''
    preprocessor = build_robust_preprocessor()
    classifier = SVC(
        C=C,
        gamma=gamma,
        kernel=kernel,
        class_weight=class_weight,
        probability=True,  # Indispensable pour predict_proba au seuil de 0.22
        random_state=42
    )
    return make_pipeline(preprocessor, classifier)


def build_pca_logistic_model(n_components=0.95, C=1.0, class_weight=None):
    '''
    Construit une pipeline Réduction de dimension (ACP) + Régression Logistique
    (cf. slides10 sur la réduction de dimension).

    L'ACP projette les variables (mises à l'échelle) sur les axes de plus forte
    variance avant la classification. Avec n_components=0.95, on conserve les
    composantes expliquant 95 % de la variance, ce qui réduit le bruit et la
    redondance (les BILL_AMT sont très corrélés) tout en restant prédictif.
    '''
    preprocessor = build_robust_preprocessor()
    # svd_solver="covariance_eigh" : seul solveur acceptant À LA FOIS une matrice
    # creuse (sortie du OneHotEncoder) ET une fraction de variance (n_components=0.95).
    # Il est aussi le plus efficace ici, car on a beaucoup plus d'exemples que de
    # variables (~18k lignes pour ~80 colonnes).
    pca = PCA(n_components=n_components, svd_solver="covariance_eigh")
    classifier = LogisticRegression(
        C=C, class_weight=class_weight, max_iter=1000, random_state=42
    )
    return make_pipeline(preprocessor, pca, classifier)

def build_xgboost_model(n_estimators=100, learning_rate=0.1, max_depth=3,
                        scale_pos_weight=1):
    """
    Pipeline XGBoost (eXtreme Gradient Boosting).
    
    XGBoost est une implémentation optimisée du Gradient Boosting avec régularisation
    L1/L2 intégrée (paramètres alpha et lambda), élagage des arbres par profondeur
    maximale et gestion native du déséquilibre via scale_pos_weight =
    n_négatifs / n_positifs (~3.5 ici).
    Les arbres n'ont pas besoin de normalisation : on utilise build_tree_preprocessor.
    """
    preprocessor = build_tree_preprocessor()
    classifier = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,  # gère le déséquilibre (~78/22 ≈ 3.5)
        eval_metric="aucpr",                # optimise le PR-AUC nativement
        use_label_encoder=False,
        random_state=42,
        n_jobs=1,
    )
    return Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', classifier) # On fige le nom de l'étape sur "xgb"
])


def build_stacking_model(class_weight="balanced"):
    """
    Pipeline Stacking (empilement de modèles, cf. cours méthodes d'ensemble).

    Niveau 0 (base learners) : Random Forest + Gradient Boosting + XGBoost.
    Ces trois modèles produisent des prédictions de probabilité qui deviennent
    les features du méta-apprenants.
    
    Niveau 1 (méta-apprenant) : Régression Logistique.
    Elle apprend à combiner les prédictions des modèles de base de façon optimale.
    La validation croisée interne (cv=5) évite le surapprentissage sur le train set.

    Les modèles de base utilisent build_tree_preprocessor (pas besoin de normalisation).
    Le méta-apprenant reçoit directement les probabilités (déjà sur [0,1]).
    """
    base_learners = [
        ("rf",  build_random_forest_model(class_weight=class_weight)),
        ("gb",  build_gradient_boost_model()),
        ("xgb", build_xgboost_model()),
    ]
    meta_learner = LogisticRegression(
        C=1.0, class_weight=class_weight, max_iter=1000, random_state=42
    )
    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    # Le stacking gère en interne ses propres pipelines (pré-traitement inclus)
    # On l'enveloppe dans un make_pipeline sans pré-traitement supplémentaire
    return stacking
def get_model_configurations(class_weight="balanced"):
    '''
    Registre central de la modélisation : associe à chaque modèle du cours
    son CONSTRUCTEUR de pipeline et son ESPACE DE RECHERCHE Optuna. C'est le cœur
    du "pipeline unique" : run_model_comparison() (cf. evaluation.py) itère sur ce
    dictionnaire pour optimiser puis comparer tous les modèles en une seule passe,
    au lieu de répéter le code pour chacun.

    Pour chaque modèle on fournit :
      - "build"   : une fonction sans argument renvoyant une pipeline NEUVE
                    (Optuna doit repartir d'un estimateur vierge à chaque essai) ;
      - "suggest" : une fonction(trial) renvoyant les hyper-paramètres tirés par
                    Optuna, sous forme {clé_pipeline: valeur}. Les clés ('svc__C',
                    etc.) correspondent au nom des étapes générées par make_pipeline,
                    si bien que study.best_params se réinjecte directement via
                    set_params() ;
      - "n_trials": (optionnel) nombre d'essais Optuna propre au modèle, pour
                    réduire le coût des modèles lents.

    On passe class_weight="balanced" uniquement aux modèles qui le supportent
    (le déséquilibre ~22 % de défauts est ainsi traité de façon homogène). Le KNN
    et le Gradient Boosting n'ont pas ce paramètre : on les laisse tels quels.

    On définit ici des plages
    continues (suggest_float en échelle log pour C / learning_rate, suggest_int
    pour les profondeurs / nombres d'arbres). Optuna (échantillonneur TPE) explore
    ces plages plus finement et plus efficacement qu'une grille exhaustive.
    '''
    return {
        "Régression Logistique": {
            "build": lambda cw=class_weight: build_logistic_model(class_weight=cw),
            "suggest": lambda trial: {
                "logisticregression__C": trial.suggest_float(
                    "logisticregression__C", 1e-3, 1e2, log=True
                ),
            },
        },
        "KNN": {
            "build": lambda: build_knn_model(),
            "suggest": lambda trial: {
                "kneighborsclassifier__n_neighbors": trial.suggest_int(
                    "kneighborsclassifier__n_neighbors", 5, 51, step=2
                ),
                "kneighborsclassifier__weights": trial.suggest_categorical(
                    "kneighborsclassifier__weights", ["uniform", "distance"]
                ),
            },
        },
        "Arbre de Décision": {
            "build": lambda cw=class_weight: build_decision_tree_model(class_weight=cw),
            "suggest": lambda trial: {
                "decisiontreeclassifier__max_depth": trial.suggest_int(
                    "decisiontreeclassifier__max_depth", 2, 20
                ),
                "decisiontreeclassifier__min_samples_leaf": trial.suggest_int(
                    "decisiontreeclassifier__min_samples_leaf", 1, 50
                ),
            },
        },
        "Random Forest": {
            "build": lambda cw=class_weight: build_random_forest_model(class_weight=cw),
            "suggest": lambda trial: {
                "randomforestclassifier__n_estimators": trial.suggest_int(
                    "randomforestclassifier__n_estimators", 100, 400, step=50
                ),
                "randomforestclassifier__max_depth": trial.suggest_int(
                    "randomforestclassifier__max_depth", 3, 20
                ),
                "randomforestclassifier__min_samples_leaf": trial.suggest_int(
                    "randomforestclassifier__min_samples_leaf", 1, 20
                ),
            },
        },
        "AdaBoost": {
            "build": lambda cw=class_weight: build_adaboost_model(class_weight=cw),
            "suggest": lambda trial: {
                "adaboostclassifier__n_estimators": trial.suggest_int(
                    "adaboostclassifier__n_estimators", 50, 300, step=50
                ),
                "adaboostclassifier__learning_rate": trial.suggest_float(
                    "adaboostclassifier__learning_rate", 0.01, 2.0, log=True
                ),
            },
        },
        "Gradient Boosting": {
            # Pas de class_weight pour GradientBoosting (paramètre inexistant)
            "build": lambda: build_gradient_boost_model(),
            "suggest": lambda trial: {
                "gradientboostingclassifier__n_estimators": trial.suggest_int(
                    "gradientboostingclassifier__n_estimators", 100, 300, step=50
                ),
                "gradientboostingclassifier__learning_rate": trial.suggest_float(
                    "gradientboostingclassifier__learning_rate", 0.01, 0.3, log=True
                ),
                "gradientboostingclassifier__max_depth": trial.suggest_int(
                    "gradientboostingclassifier__max_depth", 2, 5
                ),
            },
        },
        "ACP + Logistique": {
            "build": lambda cw=class_weight: build_pca_logistic_model(class_weight=cw),
            "suggest": lambda trial: {
                "pca__n_components": trial.suggest_float("pca__n_components", 0.80, 0.99),
                "logisticregression__C": trial.suggest_float(
                    "logisticregression__C", 1e-2, 1e2, log=True
                ),
            },
        },
        "Stacking": {
            "build": lambda cw=class_weight: build_stacking_model(class_weight=cw),
            "suggest": lambda trial: {
                 # On optimise uniquement le méta-apprenant (les base learners
                # sont déjà bons avec leurs hyperparamètres par défaut)
                "final_estimator__C": trial.suggest_float("final_estimator__C", 1e-3, 1e2, log=True),
            },
            "n_trials": 25, # Aligné sur le budget commun de 25 essais Optuna.
        },
          "SVM": {
            "build": lambda cw=class_weight: build_svm_model(class_weight=cw),
            "suggest": lambda trial: {
                "svc__C": trial.suggest_float("svc__C", 0.1, 100, log=True),
                "svc__gamma": trial.suggest_categorical("svc__gamma", ["scale", "auto"]),
            },
            # La SVM à noyau RBF est en O(n²), donc coûteuse sur ~18k exemples × cv
            # plis, mais on l'aligne sur le budget commun de 25 essais Optuna.
            "n_trials": 25,
        },
    }

'''
Corriger puis ajouter dans get_model_configurations ou évaluer à part pour avoir les hyper params optimisés
   "XGBoost": {
            "build": lambda: build_xgboost_model(),
            "suggest": lambda trial: {
                "xgb__n_estimators": trial.suggest_int("xgb__n_estimators", 100, 400, step=50),
                "xgb__learning_rate": trial.suggest_float("xgb__learning_rate", 0.01, 0.3, log=True),
                "xgb__max_depth": trial.suggest_int("xgb__max_depth", 2, 8),
                "xgb__scale_pos_weight": trial.suggest_float("xgb__scale_pos_weight", 1.0, 5.0),
            },
        },
'''
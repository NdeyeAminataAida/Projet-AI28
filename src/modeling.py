from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from src.preprocessing import build_tree_preprocessor, build_logistic_preprocessor

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

def build_svm_model(kernel='linear', C=1.0, class_weight='balanced'):
    """
    Pipeline SVM (Support Vector Machine).
    Le SVM étant basé sur des calculs de marges géométriques,
    le préprocesseur linéaire avec normalisation est obligatoire.
    """
    preprocessor = build_logistic_preprocessor()
    classifier = SVC(
        kernel=kernel, 
        C=C, 
        class_weight=class_weight, 
        probability=True,  # Indispensable pour calculer la probabilité et la PR-AUC
        random_state=42
    )
    return make_pipeline(preprocessor, classifier)

def build_knn_model(n_neighbors=5):
    """
    Pipeline KNN (K-Nearest Neighbors).
    Modèle basé sur les distances, normalisation obligatoire.
    """
    preprocessor = build_logistic_preprocessor()
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    return make_pipeline(preprocessor, classifier)
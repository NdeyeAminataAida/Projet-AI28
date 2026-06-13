from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

from src.preprocessing import build_tree_preprocessor

'''
On définit ici les fonctions qui construisent les modèles.
Chaque fonction renvoie une pipeline complète (pré-traitement + modèle)
prête à être passée à train_and_evaluate() de evaluation.py.
'''

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
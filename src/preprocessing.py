from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (OneHotEncoder,StandardScaler,RobustScaler)

'''
On définit des variables globales qu'on va réutiliser dans les différentes fonctions de pré traitement avec les pipelines
'''
# Variables catégorielles
CATEGORICAL_FEATURES = [ "SEX", "EDUCATION", "MARRIAGE"]

# Historique des retards
PAY_FEATURES = [ "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Montants facturés
BILL_FEATURES = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]

# Montants remboursés
PAY_AMT_FEATURES = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

# Variables numériques
NUMERIC_FEATURES = (["LIMIT_BAL", "AGE"]+ BILL_FEATURES+ PAY_AMT_FEATURES)

def build_logistic_preprocessor():
    '''
    Pré-processeur "à distance" : encode les catégorielles en disjonctif et
    met à l'échelle (StandardScaler) les variables numériques.

    Utilisé par tous les modèles sensibles aux échelles / aux distances :
    régression logistique, SVM (slides7) et KNN (notes cours 9), ainsi que
    l'ACP (slides10) qui repose sur la variance des variables.
    '''
    # handle_unknown="ignore" : indispensable en validation croisée, car certaines
    # modalités rares (ex. PAY_x = 8) peuvent être absentes d'un pli d'entraînement ;
    # elles sont alors encodées en zéros plutôt que de lever une erreur.
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        (OneHotEncoder(drop="first", handle_unknown="ignore"), PAY_FEATURES),
        (StandardScaler(), NUMERIC_FEATURES),
    )
    return preprocessor

def build_robust_preprocessor():
    '''
    Variante du pré-processeur "à distance" utilisant RobustScaler au lieu de
    StandardScaler pour les variables numériques.

    RobustScaler centre sur la médiane et met à l'échelle via l'écart interquartile :
    il est beaucoup moins sensible aux valeurs extrêmes que StandardScaler, ce qui
    est utile ici car les montants (BILL_AMT, PAY_AMT, LIMIT_BAL) présentent de
    fortes valeurs aberrantes.
    '''
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        (OneHotEncoder(drop="first", handle_unknown="ignore"), PAY_FEATURES),
        (RobustScaler(), NUMERIC_FEATURES),
    )
    return preprocessor

def build_tree_preprocessor():

    # Idem : on ignore les modalités inconnues pour fiabiliser la validation croisée
    preprocessor = make_column_transformer(
            (OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        remainder="passthrough"
    )
    return preprocessor
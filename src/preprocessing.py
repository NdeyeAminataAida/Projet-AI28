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
NUMERIC_FEATURES = (["LIMIT_BAL", "AGE"]+ PAY_FEATURES+ BILL_FEATURES+ PAY_AMT_FEATURES)

def build_logistic_preprocessor():
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(drop='first'),PAY_FEATURES),
    )
    return preprocessor

def build_robust_preprocessor():

    preprocessor = make_column_transformer(
            (OneHotEncoder(drop="first"),CATEGORICAL_FEATURES),
            (RobustScaler(),NUMERIC_FEATURES),
            (OneHotEncoder(drop='first'),PAY_FEATURES),
    )
    return preprocessor

def build_tree_preprocessor():

    preprocessor = make_column_transformer(
            (OneHotEncoder(drop="first"),CATEGORICAL_FEATURES),
        remainder="passthrough"
    )
    return preprocessor
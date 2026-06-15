import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_dataset_info(df):
    """Affiche les dimensions du jeu de données et le résumé des types."""
    print("--- Dimensions du jeu de données ---")
    print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}\n")
    print("--- Informations sur les types et valeurs non nulles ---")
    df.info()


def describe_numeric_variables(df):
    """Retourne les statistiques descriptives des variables numériques."""
    return df.describe()


def plot_missing_values(df):
    """Affiche la heatmap des données manquantes et le taux précis par variable."""
    missing_data = df.isna()

    plt.figure(figsize=(10, 5))
    sns.heatmap(missing_data, cbar=False)
    plt.title("Visualisation des valeurs manquantes")
    plt.show()

    miss_rates = (missing_data.sum() / df.shape[0]) * 100
    miss_rates = miss_rates.sort_values(ascending=True)
    print("--- Taux de valeurs manquantes (en %) ---")
    print(miss_rates)


def plot_correlation_matrix(df):
    """Affiche la grande matrice de corrélation globale des variables numériques."""
    corrmat = df.corr(numeric_only=True)

    plt.figure(figsize=(30, 30))
    sns.heatmap(corrmat, cmap="coolwarm", square=True, annot=True)
    plt.title("Matrice de corrélation globale", fontsize=20)
    plt.show()


def plot_target_correlations(df, target):
    """Visualise les coefficients de corrélation avec la variable cible."""
    corrmat = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 5))
    corrmat[target].sort_values(ascending=True)[:-1].plot(kind="barh")
    plt.title("Corrélation entre les variables et la cible")
    plt.tight_layout()
    plt.show()


def plot_numeric_vs_target(df, columns, target):
    """Génère des boxplots pour analyser les variables numériques selon la cible."""
    for col in columns:
        if col in df.columns and col != target:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=target, y=col, data=df, palette="Set2")
            plt.title(f"Distribution de {col} selon le statut de défaut")
            plt.xlabel(f"{target} (0 = Régulier, 1 = Défaut)")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()


def plot_categorical_vs_target(df, columns, target):
    """Génère des countplots pour analyser les variables catégorielles selon la cible."""
    for col in columns:
        if col in df.columns and col != target:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=col, hue=target, data=df, palette="Set2")
            plt.title(f"Répartition du défaut de paiement selon {col}")
            plt.xlabel(col)
            plt.ylabel("Nombre de clients")
            plt.tight_layout()
            plt.show()
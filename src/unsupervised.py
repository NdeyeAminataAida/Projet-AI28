import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.preprocessing import build_logistic_preprocessor

'''
Méthodes non supervisées vues en cours, utilisées ici à des fins exploratoires
(et non comme prédicteurs du défaut) :
  - Réduction de dimension par ACP (slides10)
  - Clustering par K-means (slides11)

Toutes deux reposent sur des variables mises à l'échelle : on réutilise donc le
pré-processeur "à distance" (build_logistic_preprocessor) pour rester cohérent
avec le reste du pipeline.
'''


def plot_pca_projection(X, y, target_name="Défaut"):
    '''
    Projette les données (pré-traitées et mises à l'échelle) sur les deux premières
    composantes principales (cf. slides10) et les colore selon la cible.

    Permet de visualiser en 2D si les clients en défaut se séparent des autres,
    et d'estimer la part de variance capturée par ces deux axes (titres des axes).
    '''
    # Même pré-traitement que les modèles à distance, puis ACP à 2 composantes
    preprocessor = build_logistic_preprocessor()
    X_scaled = preprocessor.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        components[:, 0], components[:, 1],
        c=y, cmap="coolwarm", alpha=0.3, s=10
    )
    plt.colorbar(scatter, label=f"{target_name} (0 = Régulier, 1 = Défaut)")
    plt.xlabel(f"Composante 1 ({var_ratio[0] * 100:.1f} % de variance)")
    plt.ylabel(f"Composante 2 ({var_ratio[1] * 100:.1f} % de variance)")
    plt.title("Projection ACP des clients (2 premières composantes)")
    plt.tight_layout()
    plt.show()

    print(f"Variance expliquée par les 2 axes : {var_ratio.sum() * 100:.1f} %")
    return pca


def analyze_kmeans_clusters(X, y, n_clusters=4):
    '''
    Segmente les clients par K-means (cf. slides11) sur les variables mises à l'échelle,
    puis croise les clusters obtenus avec le taux de défaut réel.

    Objectif exploratoire : voir si l'algorithme retrouve, sans connaître la cible,
    des groupes de clients aux profils de risque différents. On affiche le taux de
    défaut par cluster (un cluster très au-dessus de la moyenne ~22 % est un segment
    à risque).
    '''
    preprocessor = build_logistic_preprocessor()
    X_scaled = preprocessor.fit_transform(X)

    # n_init=10 pour limiter la sensibilité à l'initialisation aléatoire des centroïdes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Croisement cluster <-> taux de défaut réel
    analyse = pd.DataFrame({"cluster": clusters, "defaut": np.asarray(y)})
    resume = analyse.groupby("cluster")["defaut"].agg(
        effectif="size", taux_defaut="mean"
    )
    resume["taux_defaut"] = (resume["taux_defaut"] * 100).round(1)

    print(f"--- Taux de défaut par cluster (K-means, k={n_clusters}) ---")
    print(resume)

    plt.figure(figsize=(7, 4))
    plt.bar(resume.index.astype(str), resume["taux_defaut"], color="indianred")
    plt.axhline(
        np.asarray(y).mean() * 100, linestyle="--", color="grey",
        label=f"Taux moyen ({np.asarray(y).mean() * 100:.1f} %)"
    )
    plt.xlabel("Cluster")
    plt.ylabel("Taux de défaut (%)")
    plt.title(f"Profil de risque des clusters K-means (k={n_clusters})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return resume

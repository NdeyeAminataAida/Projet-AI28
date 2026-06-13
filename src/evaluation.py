from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Entraîne et évalue un modèle sklearn.
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("--------- Métriques pour TRAIN ---------")
    print(classification_report(y_train, y_train_pred))

    print("--------- Métriques pour TEST ---------")
    print(classification_report(y_test, y_test_pred))

    return model


def plot_confusion_matrix(y_true, y_pred, title="Matrice de confusion"):
    """
    Affiche une matrice de confusion avec heatmap seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(title)

    plt.show()

    return cm
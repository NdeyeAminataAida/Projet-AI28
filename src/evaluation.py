from sklearn.metrics import classification_report

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Entraîne et évalue un modèle sklearn.
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("=== TRAIN ===")
    print(classification_report(y_train, y_train_pred))

    print("=== TEST ===")
    print(classification_report(y_test, y_test_pred))

    return model
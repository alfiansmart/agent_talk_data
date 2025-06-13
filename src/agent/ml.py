import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error


def regression(df: pd.DataFrame, target: str):
    """Perform a simple linear regression."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse


def classification(df: pd.DataFrame, target: str):
    """Perform a logistic regression classification."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc


def clustering(df: pd.DataFrame, k: int):
    """KMeans clustering."""
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(df)
    return model, model.labels_

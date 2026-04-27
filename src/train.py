import os
import joblib
from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/movie_rating_model.pkl")

    return model

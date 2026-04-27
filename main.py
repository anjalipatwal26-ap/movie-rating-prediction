from src.preprocess import load_and_preprocess
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    data_path = "data/imdb_movies.csv"

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    mae, mse, r2 = evaluate_model(model, X_test, y_test)

    print("\nModel Performance:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")


if __name__ == "__main__":
    main()

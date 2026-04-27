import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.txt", "w") as f:
        f.write("Movie Rating Prediction Metrics\n")
        f.write("-------------------------------\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Actual vs Predicted Ratings")
    plt.savefig("outputs/regression_plot.png")
    plt.close()

    return mae, mse, r2

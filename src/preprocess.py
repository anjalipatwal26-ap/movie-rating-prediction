import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(data_path):

    # Load dataset
    df = pd.read_csv(data_path, encoding="latin-1")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop rows where Rating is missing
    df = df.dropna(subset=["Rating"])

    # Fill missing values
    df = df.fillna("Unknown")

    # Clean Year column (remove brackets if present)
    df["Year"] = df["Year"].astype(str).str.extract(r'(\d{4})')
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Clean Duration (remove ' min')
    df["Duration"] = df["Duration"].astype(str).str.replace(" min", "", regex=False)
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

    # Convert Votes to numeric (remove commas)
    df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

    # Drop remaining missing values
    df = df.dropna()

    # Encode categorical columns
    categorical_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Name"]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Features & Target
    X = df.drop("Rating", axis=1)
    y = df["Rating"]

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test




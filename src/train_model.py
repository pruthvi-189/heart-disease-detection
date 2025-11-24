import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'heart_model.joblib')

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Convert classes >1 to 1 (binary)
    if df['target'].max() > 1:
        df['target'] = (df['target'] > 0).astype(int)
    return df

def main():
    df = load_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [5, 10, None],
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc")
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC AUC:", roc_auc_score(y_test, y_proba))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

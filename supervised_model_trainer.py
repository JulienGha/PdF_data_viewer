import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from charset_normalizer import from_path


def main():

    result = from_path("zero_shot_predictions.csv").best()
    print(result.encoding)

    df = pd.read_csv(
        "zero_shot_predictions.csv",
        sep=";",
        encoding="iso8859_10"
    )
    print("Columns found:", df.columns.tolist())

    df = df[df["PredictedLabel"].notna() & (df["PredictedLabel"].str.strip() != "")]
    print(f"Using {len(df)} labeled examples spread over {df['PredictedLabel'].nunique()} classes.")

    texts = (df["EmailContent"] + df["Subject"]).tolist()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    X = model.encode(texts, convert_to_numpy=True)
    y = df["PredictedLabel"].values

    counts = pd.Series(y).value_counts()
    if counts.min() >= 2:
        strat = y
        print("Stratifying split by label.")
    else:
        strat = None
        print("Some classes have only 1 example—doing a random split (no stratification).")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        stratify=strat,
        test_size=0.1,
        random_state=42
    )

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n=== Classification Report ===\n")
    print(classification_report(y_val, y_pred))

    joblib.dump(model, "embedder.joblib")
    joblib.dump(clf,   "email_classifier.joblib")
    print("\n✅ Models saved to embedder.joblib and email_classifier.joblib")

if __name__ == "__main__":
    main()

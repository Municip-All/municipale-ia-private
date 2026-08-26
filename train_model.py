from pathlib import Path

import structlog
import pandas as pd
import numpy as np
import joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

log = structlog.get_logger("municipall.training")

ART = Path("artifacts")
DATA_CSV = ART / "preprocessed.csv"
TFIDF_PATH = ART / "tfidf.joblib"
MODEL_PATH = ART / "model_rf.joblib"
METRICS_PATH = ART / "metrics.json"
ENC_PATH = ART / "geo_onehot.joblib"

def build_features(df: pd.DataFrame):
    tfidf = joblib.load(TFIDF_PATH)
    X_text = tfidf.transform(df["description"].values)
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_geo = enc.fit_transform(df[["geo_bucket"]])
    joblib.dump(enc, ENC_PATH)
    X_hour = csr_matrix(df[["hour"]].values)
    X = hstack([X_text, X_geo, X_hour], format="csr")
    return X, tfidf, enc

def main():
    df = pd.read_csv(DATA_CSV)
    X, tfidf, enc = build_features(df)
    y = df["type"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(clf, MODEL_PATH)

    metrics = {"accuracy": acc, "f1_macro": f1m, "report": report}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info("training_done", accuracy=f"{acc:.4f}", f1_macro=f"{f1m:.4f}", model_path=str(MODEL_PATH))

if __name__ == "__main__":
    main()

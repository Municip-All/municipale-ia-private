######
#   ALKAYA MEHMET
#   EPITECH 2025
#   PROJET EDP
#####

"""
Data preprocessing for Municip'All IA
- Loads CSV (or generates synthetic if none)
- Cleans text
- Creates features (TF-IDF text, geo bucket, hour)
- Saves artifacts: preprocessed.csv, tfidf.joblib
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from utils import geo_bucket

RAW_PATH = Path("data/raw_signalements.csv")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "dechets_sauvages", "voirie", "animal_errant",
    "mobilier_urbain", "objet_abandonne", "incivilite", "espace_vert"
]

def generate_synthetic(n=20000, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    texts = []
    lats = 49.25 + np.random.rand(n) * 0.05
    lons = 2.40 + np.random.rand(n) * 0.05
    hours = np.random.randint(0, 24, size=n)
    cats = np.random.choice(CATEGORIES, size=n, p=[0.22,0.20,0.1,0.14,0.14,0.12,0.08])
    templates = {
        "dechets_sauvages": ["Tas de déchets", "Poubelles débordent", "Déchets au sol"],
        "voirie": ["Nid-de-poule", "Trottoir cassé", "Signalisation abîmée"],
        "animal_errant": ["Chien errant", "Chat blessé", "Animal sans maître"],
        "mobilier_urbain": ["Banc cassé", "Abribus brisé", "Barrière tordue"],
        "objet_abandonne": ["Vélo abandonné", "Caddie dans la rue", "Trottinette laissée"],
        "incivilite": ["Tapage nocturne", "Agression verbale", "Attroupement bruyant"],
        "espace_vert": ["Arbre tombé", "Haie non taillée", "Parterre dégradé"]
    }
    for i in range(n):
        cat = cats[i]
        txt = np.random.choice(templates[cat])
        street = np.random.choice(["rue Victor Hugo", "avenue des Lilas", "place du Marché"])
        texts.append(f"{txt} {street}")
    df = pd.DataFrame({
        "description": texts,
        "lat": lats,
        "lon": lons,
        "hour": hours,
        "type": cats
    })
    return df

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-zàâäèéêëîïôöùûüç0-9\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
    else:
        df = generate_synthetic()

    # Basic cleaning
    df["description"] = df["description"].astype(str).apply(normalize_text)
    df["hour"] = df["hour"].fillna(0).astype(int)
    df["geo_bucket"] = [geo_bucket(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    df = df.dropna(subset=["description", "type"])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df["description"].values)

    # Persist artifacts
    joblib.dump(tfidf, OUT_DIR / "tfidf.joblib")

    # Save preprocessed tabular for later merge in training
    # Note: we store text separately to avoid huge CSVs; training script will re-load tfidf for matrix.
    keep_cols = ["description", "lat", "lon", "hour", "geo_bucket", "type"]
    df[keep_cols].to_csv(OUT_DIR / "preprocessed.csv", index=False)

    print("✅ Preprocessing done.")
    print(f"- Saved TFIDF to {OUT_DIR/'tfidf.joblib'}")
    print(f"- Saved preprocessed CSV to {OUT_DIR/'preprocessed.csv'}")

if __name__ == "__main__":
    main()

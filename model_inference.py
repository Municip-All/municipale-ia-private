from pathlib import Path
import joblib
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd

ART = Path("artifacts")
TFIDF_PATH = ART / "tfidf.joblib"
MODEL_PATH = ART / "model_rf.joblib"
ENC_PATH = ART / "geo_onehot.joblib"

class Predictor:
    def __init__(self):
        self.tfidf = joblib.load(TFIDF_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.enc = joblib.load(ENC_PATH)

    def predict(self, description: str, geo_bucket: str, hour: int):
        X_text = self.tfidf.transform([description or ""])
        geo_df = pd.DataFrame({"geo_bucket": [geo_bucket]})
        X_geo = self.enc.transform(geo_df)
        X_hour = csr_matrix([[hour]])
        X = hstack([X_text, X_geo, X_hour], format="csr")
        proba = self.model.predict_proba(X)[0]
        cls_idx = int(np.argmax(proba))
        cls = self.model.classes_[cls_idx]
        return {"pred": cls, "proba": float(proba[cls_idx])}

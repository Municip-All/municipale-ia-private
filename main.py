from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import logging

from data_preprocessing import main as preprocessing_main
from model_inference import Predictor
from train_model import main as training_main
from utils import geo_bucket

logger = logging.getLogger("municipall.main")

ART = Path("artifacts")
DATA_CSV = ART / "preprocessed.csv"
TFIDF_PATH = ART / "tfidf.joblib"
MODEL_PATH = ART / "model_rf.joblib"
ENC_PATH = ART / "geo_onehot.joblib"
METRICS_PATH = ART / "metrics.json"


def ensure_preprocessing(force: bool) -> None:
    if force or not DATA_CSV.exists() or not TFIDF_PATH.exists():
        logger.info("preprocessing_start")
        preprocessing_main()
    else:
        logger.info("preprocessing_skipped")


def ensure_training(force: bool) -> None:
    if force or not MODEL_PATH.exists() or not ENC_PATH.exists():
        logger.info("training_start")
        training_main()
    else:
        logger.info("training_skipped")


def run_demo_inference(description: str, lat: float, lon: float, hour: int) -> dict:
    predictor = Predictor()
    bucket = geo_bucket(lat, lon)
    result = predictor.predict(description=description, geo_bucket=bucket, hour=hour)
    result["geo_bucket"] = bucket
    return result


def clean_artifacts() -> None:
    if ART.exists():
        shutil.rmtree(ART)
        logger.info("artifacts_cleaned")
    else:
        logger.info("artifacts_empty")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lancer l'ensemble du pipeline IA Municip'All."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force le recalcul du prétraitement et la réentraînement du modèle.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Saute l'inférence de démonstration après l'entraînement.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Supprime les artefacts puis arrête le script.",
    )
    parser.add_argument(
        "--description",
        default="Tas de déchets rue Victor Hugo",
        help="Description texte utilisée pour la démo d'inférence.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=49.265,
        help="Latitude pour la démo d'inférence.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=2.435,
        help="Longitude pour la démo d'inférence.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=10,
        help="Heure du signalement pour la démo d'inférence.",
    )
    args = parser.parse_args()

    if args.clean:
        clean_artifacts()
        return

    ART.mkdir(parents=True, exist_ok=True)

    ensure_preprocessing(force=args.force)
    ensure_training(force=args.force)

    if args.skip_demo:
        logger.info("pipeline_done_skip_demo")
        return

    logger.info("demo_inference_start")
    demo = run_demo_inference(
        description=args.description, lat=args.lat, lon=args.lon, hour=args.hour
    )

    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    logger.info(
        "pipeline_done pred=%s proba=%s geo=%s acc=%s f1=%s",
        demo["pred"],
        f"{demo['proba']:.3f}",
        demo["geo_bucket"],
        metrics.get("accuracy", "NA"),
        metrics.get("f1_macro", "NA"),
    )


if __name__ == "__main__":
    main()

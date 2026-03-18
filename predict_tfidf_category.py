#!/usr/bin/env python3
"""Predict a paper category from title and abstract using a saved TF-IDF model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib


class TfidfPaperCategorizer:
    """Load trained TF-IDF artifacts and run single-example inference."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.vectorizer = self._load_artifact("tfidf_vectorizer.joblib")
        self.model = self._load_artifact("tfidf_logreg.joblib")
        self.label_encoder = self._load_artifact("label_encoder.joblib")

    def _load_artifact(self, artifact_name: str) -> Any:
        artifact_path = self.model_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifact: {artifact_path}")
        return joblib.load(artifact_path)

    @staticmethod
    def build_model_text(title: str, abstract: str) -> str:
        # Matches the training pipeline from notebook 05: title + space + abstract.
        return f"{(title or '').strip()} {(abstract or '').strip()}".strip()

    def predict(self, title: str, abstract: str, top_k: int = 3) -> dict[str, Any]:
        model_text = self.build_model_text(title, abstract)
        if not model_text:
            raise ValueError("Title and abstract cannot both be empty.")

        features = self.vectorizer.transform([model_text])
        pred_idx = self.model.predict(features)
        pred_label = self.label_encoder.inverse_transform(pred_idx)[0]

        result: dict[str, Any] = {"predicted_label": str(pred_label)}

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            ranked = sorted(
                zip(self.label_encoder.classes_, probs),
                key=lambda item: item[1],
                reverse=True,
            )
            keep = max(1, int(top_k))
            result["top_probabilities"] = [
                {"label": str(label), "probability": float(prob)}
                for label, prob in ranked[:keep]
            ]

        return result


def parse_args() -> argparse.Namespace:
    default_model_dir = Path(__file__).resolve().parent / "models" / "tfidf"

    parser = argparse.ArgumentParser(
        description="Predict a research paper category from title and abstract."
    )
    parser.add_argument("--title", type=str, help="Paper title.")
    parser.add_argument("--abstract", type=str, help="Paper abstract.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help=f"Directory containing saved artifacts (default: {default_model_dir}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top class probabilities to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    title = args.title if args.title is not None else input("Title: ").strip()
    abstract = args.abstract if args.abstract is not None else input("Abstract: ").strip()

    predictor = TfidfPaperCategorizer(model_dir=args.model_dir)
    prediction = predictor.predict(title=title, abstract=abstract, top_k=args.top_k)

    print(f"\nPredicted category: {prediction['predicted_label']}")

    top_probs = prediction.get("top_probabilities", [])
    if top_probs:
        print("Top probabilities:")
        for row in top_probs:
            print(f"  - {row['label']}: {row['probability']:.4f}")


if __name__ == "__main__":
    main()

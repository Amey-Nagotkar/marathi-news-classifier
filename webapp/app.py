"""
Marathi News Headline Classification System - Flask Web Application
Backend Server (Phase 8.6)

Exposes RESTful prediction, dataset exploration, and statistics endpoints
calling the frozen NLP and ML inference pipeline.
"""

import os
import sys
import json
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_headline, load_model
from src.preprocessing import clean_headline
from src.explain import explain_prediction

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)

# Global dataset cache
_DATASET_DF = None
_DATASET_STATS = None

DATASET_PATH = os.path.join(BASE_DIR, "Master_Final_Marathi_News_5Label.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")

VALID_CATEGORIES = [
    "Technology & Science",
    "Politics",
    "Business",
    "Sports",
    "Entertainment"
]


def get_dataset() -> pd.DataFrame:
    """Loads and caches the master dataset."""
    global _DATASET_DF
    if _DATASET_DF is None:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Master dataset not found at {DATASET_PATH}")
        _DATASET_DF = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    return _DATASET_DF


def get_dataset_statistics() -> Dict[str, Any]:
    """Computes and caches dataset summary metrics."""
    global _DATASET_STATS
    if _DATASET_STATS is None:
        df = get_dataset()
        total_records = len(df)
        counts = df["Label"].value_counts().to_dict()
        percentages = {cat: round((cnt / total_records) * 100, 2) for cat, cnt in counts.items()}

        train_count = 11298
        test_count = 2825
        if os.path.exists(TRAIN_PATH):
            train_count = sum(1 for _ in open(TRAIN_PATH, encoding="utf-8-sig")) - 1
        if os.path.exists(TEST_PATH):
            test_count = sum(1 for _ in open(TEST_PATH, encoding="utf-8-sig")) - 1

        _DATASET_STATS = {
            "total_records": total_records,
            "total_categories": df["Label"].nunique(),
            "category_counts": counts,
            "category_percentages": percentages,
            "train_records": train_count,
            "test_records": test_count,
            "categories": VALID_CATEGORIES
        }
    return _DATASET_STATS


# Pre-warm model cache at startup
try:
    load_model()
except Exception as e:
    app.logger.warning(f"Warning during pre-warming model: {e}")


# ============================================================
# Page Routes
# ============================================================

@app.route("/")
def index():
    """Home / Headline Classification Page."""
    return render_template("index.html", page="home")


@app.route("/dataset")
def dataset_page():
    """Dataset Explorer Page."""
    stats = get_dataset_statistics()
    return render_template("dataset.html", page="dataset", stats=stats)


@app.route("/model")
def model_page():
    """Model Performance & Architecture Page."""
    return render_template("model.html", page="model")


@app.route("/about")
def about_page():
    """Academic Project Overview Page."""
    return render_template("about.html", page="about")


# ============================================================
# API Endpoints
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Headline Classification API endpoint.
    Accepts JSON: {"headline": "..."}
    Returns predicted category, model probability, top-3 predictions,
    all class probabilities, and preprocessed text.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        # Fallback to form data
        headline = request.form.get("headline", "")
    else:
        headline = data.get("headline", "")

    if headline is None or not str(headline).strip():
        return jsonify({
            "success": False,
            "error": "Please enter a Marathi news headline."
        }), 400

    try:
        res = predict_headline(str(headline))
        explain_res = explain_prediction(str(headline))
        response_data = {
            "success": True,
            "headline": res["text"],
            "predicted_category": res["predicted_label"],
            "model_probability": res["predicted_probability"],
            "top_predictions": [
                {
                    "category": item["label"],
                    "probability": item["probability"]
                }
                for item in res["top_predictions"]
            ],
            "all_probabilities": res["all_class_probabilities"],
            "preprocessed_text": res["cleaned_text"],
            "explainability": {
                "top_features": explain_res["top_contributing_features"],
                "disclaimer": explain_res["disclaimer"],
                "features_count": explain_res["features_present_count"]
            }
        }
        return jsonify(response_data), 200

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": "Unable to classify the headline. Please try again."
        }), 500


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """
    Dedicated Explainability API endpoint.
    Accepts JSON: {"headline": "..."}
    Returns feature contributions for the predicted category.
    """
    data = request.get_json(silent=True) or request.form
    headline = data.get("headline", "") if data else ""

    if headline is None or not str(headline).strip():
        return jsonify({
            "success": False,
            "error": "Please enter a Marathi news headline."
        }), 400

    try:
        res = explain_prediction(str(headline))
        return jsonify({
            "success": True,
            **res
        }), 200
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        app.logger.error(f"Explain error: {e}")
        return jsonify({
            "success": False,
            "error": "Unable to compute feature explainability."
        }), 500


@app.route("/api/random-headline", methods=["GET"])
def api_random_headline():
    """
    Returns a real headline sampled from Master_Final_Marathi_News_5Label.csv.
    Supports ?category=Sports or ?category=all.
    """
    try:
        df = get_dataset()
        category = request.args.get("category", "all").strip()

        if category and category.lower() != "all":
            subset = df[df["Label"].str.lower() == category.lower()]
            if subset.empty:
                # If invalid category provided, return 400
                return jsonify({
                    "success": False,
                    "error": f"Category '{category}' not found. Valid: {VALID_CATEGORIES}"
                }), 400
            sample_row = subset.sample(n=1).iloc[0]
        else:
            sample_row = df.sample(n=1).iloc[0]

        text_col = "Text" if "Text" in sample_row else "Headline"
        return jsonify({
            "success": True,
            "headline": str(sample_row[text_col]),
            "category": str(sample_row["Label"])
        }), 200

    except Exception as e:
        app.logger.error(f"Random headline error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to sample headline from dataset."
        }), 500


@app.route("/api/dataset-stats", methods=["GET"])
def api_dataset_stats():
    """
    Returns dataset statistics (record counts, percentages, splits).
    """
    try:
        stats = get_dataset_statistics()
        return jsonify({
            "success": True,
            "total_records": stats["total_records"],
            "total_categories": stats["total_categories"],
            "category_counts": stats["category_counts"],
            "category_percentages": stats["category_percentages"],
            "train_records": stats["train_records"],
            "test_records": stats["test_records"]
        }), 200
    except Exception as e:
        app.logger.error(f"Dataset stats error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to compute dataset statistics."
        }), 500


@app.route("/api/category/<path:category_name>", methods=["GET"])
def api_category_explorer(category_name: str):
    """
    Returns category metrics and 5 real sample headlines.
    """
    try:
        df = get_dataset()
        category_name = category_name.strip()

        # Find matching category case-insensitively
        matches = [c for c in VALID_CATEGORIES if c.lower() == category_name.lower()]
        if not matches:
            return jsonify({
                "success": False,
                "error": f"Invalid category. Valid options: {VALID_CATEGORIES}"
            }), 404

        matched_cat = matches[0]
        subset = df[df["Label"] == matched_cat]
        record_count = int(len(subset))
        total_records = int(len(df))
        percentage = round((record_count / total_records) * 100, 2)

        sample_count = min(5, record_count)
        text_col = "Text" if "Text" in subset.columns else "Headline"
        examples = subset.sample(n=sample_count, random_state=42)[text_col].tolist()
        
        # Calculate genuine average headline word length for category
        word_lengths = subset[text_col].astype(str).str.split().str.len()
        avg_headline_length = round(float(word_lengths.mean()), 1)

        return jsonify({
            "success": True,
            "category": matched_cat,
            "record_count": record_count,
            "percentage": percentage,
            "avg_headline_length": avg_headline_length,
            "examples": examples
        }), 200

    except Exception as e:
        app.logger.error(f"Category explorer error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve category details."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"Marathi News Classifier - Flask Server")
    print(f"Server running locally at: http://127.0.0.1:{port}")
    print(f"100% Local Inference | No External APIs Required")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=True)

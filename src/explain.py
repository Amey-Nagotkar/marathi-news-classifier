"""
Explainability & Interpretability Module for Marathi News Headline Classification
Part of Marathi News Headline Classification System (Phase 8.6.1).

Derives genuine model-level feature contributions for linear classification:
Contribution_i = TFIDF_value_i * Logistic_Regression_Coefficient_i

Only observed terms present in the input headline are evaluated.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_headline
from src.predict import load_model

EXPLAINABILITY_DISCLAIMER = (
    "These terms are the highest TF-IDF feature contributions for the predicted class "
    "in this headline. They indicate which observed terms had the strongest positive "
    "influence on the model score."
)


def explain_prediction(
    text: str,
    vectorizer: Optional[Any] = None,
    classifier: Optional[Any] = None,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Computes genuine feature-level contributions for the predicted category
    using the trained TF-IDF vectorizer and Logistic Regression coefficients.

    Parameters
    ----------
    text : str
        Raw input headline.
    vectorizer : TfidfVectorizer, optional
        Pre-loaded vectorizer. Loaded from disk if None.
    classifier : LogisticRegression, optional
        Pre-loaded classifier. Loaded from disk if None.
    top_n : int, default=5
        Number of top contributing features to return.

    Returns
    -------
    dict
        Structured explainability result containing predicted category,
        top contributing terms, their exact contribution scores, and disclaimer.
    """
    if text is None:
        raise ValueError("Please enter a Marathi news headline (input was None).")
    if not isinstance(text, str):
        raise ValueError(f"Headline must be text/string, received {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Please enter a Marathi news headline (input is empty).")

    cleaned_text = clean_headline(text, lowercase_english=True)
    if not cleaned_text.strip():
        raise ValueError("Please enter a valid Marathi news headline (headline contains no readable text).")

    if vectorizer is None or classifier is None:
        vectorizer, classifier = load_model()

    features_sparse = vectorizer.transform([cleaned_text])
    probabilities = classifier.predict_proba(features_sparse)[0]
    classes = classifier.classes_

    best_idx = int(np.argmax(probabilities))
    predicted_category = str(classes[best_idx])
    model_probability = round(float(probabilities[best_idx]), 4)

    # Extract non-zero features (terms actually present in the headline)
    nonzero_indices = features_sparse.nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[best_idx]

    contributions: List[Dict[str, Any]] = []
    for idx in nonzero_indices:
        tfidf_val = float(features_sparse[0, idx])
        coef_val = float(coefs[idx])
        contrib = tfidf_val * coef_val
        contributions.append({
            "term": str(feature_names[idx]),
            "contribution": round(contrib, 4),
            "tfidf": round(tfidf_val, 4),
            "coefficient": round(coef_val, 4),
            "is_positive": contrib >= 0
        })

    # Sort descending by contribution
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    # Top positive contributing features
    top_positive = [c for c in contributions if c["contribution"] > 0][:top_n]
    if not top_positive and contributions:
        # If all contributions happen to be <= 0, take highest available
        top_positive = contributions[:top_n]

    return {
        "headline": text,
        "preprocessed_text": cleaned_text,
        "predicted_category": predicted_category,
        "model_probability": model_probability,
        "features_present_count": len(nonzero_indices),
        "top_contributing_features": top_positive,
        "all_present_features": contributions,
        "disclaimer": EXPLAINABILITY_DISCLAIMER
    }

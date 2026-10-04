"""
Inference & Prediction Module for Marathi News Headline Classification
Part of Marathi News Headline Classification System (Phase 7).

Provides clean, reusable functions to load serialized production artifacts
and perform real-time classification on single headlines or batches.

Inference Pipeline:
Raw User Headline
       ↓
Input Validation
       ↓
src.preprocessing.clean_headline()
       ↓
models/tfidf_vectorizer.joblib
       ↓
models/final_classifier.joblib
       ↓
predict_proba() Softmax Probabilities
       ↓
Structured Output (Predicted Label, Confidence %, Top-3 Predictions)
"""

import os
import sys
from typing import Dict, Any, List, Union, Optional
import joblib
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_headline

DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models")

# Module-level cache for loaded models
_CACHED_VECTORIZER = None
_CACHED_CLASSIFIER = None


def load_model(models_dir: Optional[str] = None) -> tuple:
    """
    Loads the trained TF-IDF vectorizer and Logistic Regression classifier from disk.
    Uses cached instances if already loaded into memory.

    Parameters
    ----------
    models_dir : str, optional
        Path to the directory containing joblib artifacts.
        Defaults to project 'models/' directory.

    Returns
    -------
    tuple
        (vectorizer, classifier)
    """
    global _CACHED_VECTORIZER, _CACHED_CLASSIFIER

    if _CACHED_VECTORIZER is not None and _CACHED_CLASSIFIER is not None:
        return _CACHED_VECTORIZER, _CACHED_CLASSIFIER

    target_dir = models_dir or DEFAULT_MODELS_DIR
    vec_path = os.path.join(target_dir, "tfidf_vectorizer.joblib")
    clf_path = os.path.join(target_dir, "final_classifier.joblib")

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Missing vectorizer artifact: {vec_path}")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Missing classifier artifact: {clf_path}")

    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(clf_path)

    _CACHED_VECTORIZER = vectorizer
    _CACHED_CLASSIFIER = classifier

    return vectorizer, classifier


def predict_headline(
    text: Union[str, Any],
    vectorizer: Optional[Any] = None,
    classifier: Optional[Any] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Classifies a single Marathi news headline.

    Parameters
    ----------
    text : str
        Raw input headline.
    vectorizer : TfidfVectorizer, optional
        Pre-loaded vectorizer. Loaded from disk if None.
    classifier : LogisticRegression, optional
        Pre-loaded classifier. Loaded from disk if None.
    top_k : int, default=3
        Number of top probability classes to return.

    Returns
    -------
    dict
        Structured prediction result with confidence and top predictions.
    """
    # 1. Input Validation
    if text is None:
        raise ValueError("Please enter a Marathi news headline (input was None).")
    if not isinstance(text, str):
        raise ValueError(f"Headline must be text/string, received {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Please enter a Marathi news headline (input is empty).")

    # 2. Preprocess with Phase 2 Reusable Logic
    cleaned_text = clean_headline(text, lowercase_english=True)
    if not cleaned_text.strip():
        raise ValueError("Please enter a valid Marathi news headline (headline contains no readable text).")

    # 3. Load Models if not provided
    if vectorizer is None or classifier is None:
        vectorizer, classifier = load_model()

    # 4. TF-IDF Transformation
    features = vectorizer.transform([cleaned_text])

    # 5. Probability Estimation
    probabilities = classifier.predict_proba(features)[0]
    classes = classifier.classes_

    # 6. Rank Predictions by Probability
    sorted_indices = np.argsort(probabilities)[::-1]
    best_idx = sorted_indices[0]
    predicted_label = classes[best_idx]
    predicted_prob = float(probabilities[best_idx])

    top_predictions = []
    limit = min(top_k, len(classes))
    for idx in sorted_indices[:limit]:
        top_predictions.append({
            "label": classes[idx],
            "probability": round(float(probabilities[idx]), 4),
            "percentage": f"{probabilities[idx] * 100:.2f}%"
        })

    # Return Structured Result
    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "predicted_label": predicted_label,
        "predicted_probability": round(predicted_prob, 4),
        "confidence_percentage": f"{predicted_prob * 100:.2f}%",
        "top_predictions": top_predictions,
        "all_class_probabilities": {
            classes[i]: round(float(probabilities[i]), 4) for i in sorted_indices
        }
    }


def predict_batch(
    texts: List[str],
    vectorizer: Optional[Any] = None,
    classifier: Optional[Any] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Classifies a list of Marathi headlines efficiently in batch.
    """
    if vectorizer is None or classifier is None:
        vectorizer, classifier = load_model()

    results = []
    for text in texts:
        try:
            res = predict_headline(text, vectorizer, classifier, top_k=top_k)
            results.append(res)
        except ValueError as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    return results


if __name__ == '__main__':
    # Interactive verification and benchmark demonstrations
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 65)
    print("MARATHI NEWS CLASSIFIER — INFERENCE ENGINE VERIFICATION")
    print("=" * 65)

    vec, clf = load_model()
    print(f"[1] Loaded Artifacts Successfully:")
    print(f"    - Vectorizer vocabulary: {len(vec.vocabulary_)} features")
    print(f"    - Classifier classes:    {list(clf.classes_)}")

    test_samples = [
        ("भारताने ऑस्ट्रेलियाचा ५ विकेट्सने पराभव केला", "Sports"),
        ("नवीन 5G स्मार्टफोन भारतात लाँच, किंमत फक्त...", "Technology & Science"),
        ("मुख्यमंत्री एकनाथ शिंदे यांच्या उपस्थितीत बैठक", "Politics"),
        ("शेअर बाजारात सेन्सेक्स आणि निफ्टीमध्ये मोठी घसरण", "Business"),
        ("दिवाळीच्या मुहूर्तावर नवीन मराठी चित्रपटाचा ट्रेलर रिलीज", "Entertainment")
    ]

    print("\n[2] Running Benchmark Predictions:")
    print("-" * 65)
    for text, expected in test_samples:
        result = predict_headline(text, vec, clf)
        print(f"\nHeadline: '{text}'")
        print(f"Expected: {expected} | Predicted: {result['predicted_label']} (Confidence: {result['confidence_percentage']})")
        print("Top 3 Predictions:")
        for top in result['top_predictions']:
            print(f"  - {top['label']:<22}: {top['percentage']}")

    print("\n[3] Input Validation Handling:")
    try:
        predict_headline("   ")
    except ValueError as e:
        print(f"    [PASS] Empty string rejected gracefully: '{e}'")

    try:
        predict_headline(None)
    except ValueError as e:
        print(f"    [PASS] None input rejected gracefully: '{e}'")

    print("\n" + "=" * 65)
    print("PREDICTION MODULE VERIFICATION COMPLETE")
    print("=" * 65)

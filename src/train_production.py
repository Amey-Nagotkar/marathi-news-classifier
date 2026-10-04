"""
Production Model Training & Artifact Serialization Module
Part of Marathi News Headline Classification System (Phase 7).

Fits the finalized, approved architecture:
- Feature Extractor: TfidfVectorizer (Devanagari-safe, unigram+bigram, min_df=2, sublinear_tf=True)
- Classifier: LogisticRegression (max_iter=2000, class_weight='balanced', random_state=42)
on the complete 14,123-record master dataset for maximum vocabulary and parameter coverage.

Saves:
- models/tfidf_vectorizer.joblib
- models/final_classifier.joblib
- models/marathi_news_classifier.joblib (unified pipeline)
- models/model_metadata.json
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import load_dataset, clean_headline, preprocess_dataset, MARATHI_TOKEN_PATTERN
from src.features import create_vectorizer
from src.train import LABELS_ORDER


def train_production_artifacts(
    master_csv_path: str,
    models_dir: str
):
    """
    Trains production vectorizer and classifier on the full master dataset (14,123 records)
    and serializes the artifacts with joblib.
    """
    print("=" * 70)
    print("PHASE 7: PRODUCTION MODEL TRAINING & SERIALIZATION")
    print("=" * 70)

    # 1. Load Master Dataset
    print(f"\n[1] Loading complete master dataset from:\n    {master_csv_path}")
    df = load_dataset(master_csv_path)
    total_records = len(df)
    print(f"    Loaded records: {total_records}")
    assert total_records == 14123, f"Expected 14,123 records, got {total_records}"

    # 2. Preprocess
    print("\n[2] Applying Phase 2 text preprocessing...")
    df_clean = preprocess_dataset(df, text_column='Text', cleaned_column='Cleaned_Text')
    X_full = df_clean['Cleaned_Text']
    y_full = df_clean['Label']

    # 3. Create & Fit TF-IDF Vectorizer
    print("\n[3] Fitting production TF-IDF Vectorizer...")
    vectorizer = create_vectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        sublinear_tf=True,
        use_idf=True,
        lowercase=False
    )
    X_full_tfidf = vectorizer.fit_transform(X_full)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"    Production Vocabulary Size: {vocab_size} features (across all 14,123 records)")
    print(f"    Matrix Shape: {X_full_tfidf.shape}")

    # 4. Fit Logistic Regression Classifier
    print("\n[4] Fitting production Logistic Regression (Balanced)...")
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    classifier.fit(X_full_tfidf, y_full)
    print("    [PASS] Classifier fitted successfully.")
    print(f"    Classes: {list(classifier.classes_)}")

    # 5. Create Unified Pipeline
    # A single pipeline object combining vectorizer and classifier
    unified_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    # 6. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    clf_path = os.path.join(models_dir, "final_classifier.joblib")
    pipe_path = os.path.join(models_dir, "marathi_news_classifier.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    print(f"\n[5] Serializing production artifacts to:\n    {models_dir}")
    joblib.dump(vectorizer, vec_path, compress=3)
    joblib.dump(classifier, clf_path, compress=3)
    joblib.dump(unified_pipeline, pipe_path, compress=3)

    print(f"    - Vectorizer: {vec_path} ({os.path.getsize(vec_path)} bytes)")
    print(f"    - Classifier: {clf_path} ({os.path.getsize(clf_path)} bytes)")
    print(f"    - Pipeline:   {pipe_path} ({os.path.getsize(pipe_path)} bytes)")

    # 7. Save Model Metadata
    metadata = {
        "project": "Marathi News Headline Classification System",
        "model_architecture": "Logistic Regression with Balanced Class Weighting",
        "vectorizer": "TfidfVectorizer (Sublinear TF, IDF, Unigram + Bigram)",
        "token_pattern": MARATHI_TOKEN_PATTERN,
        "tfidf_ngram_range": [1, 2],
        "tfidf_min_df": 2,
        "tfidf_max_df": 1.0,
        "tfidf_sublinear_tf": True,
        "tfidf_use_idf": True,
        "tfidf_lowercase": False,
        "classifier_hyperparameters": {
            "max_iter": 2000,
            "random_state": 42,
            "class_weight": "balanced",
            "solver": "lbfgs"
        },
        "classes": list(classifier.classes_),
        "canonical_classes_order": LABELS_ORDER,
        "production_training_samples": total_records,
        "production_vocabulary_size": vocab_size,
        "evaluation_benchmark": {
            "test_sample_size": 2825,
            "test_accuracy": 0.8984,
            "test_macro_f1": 0.8675,
            "test_weighted_f1": 0.8984,
            "per_class_f1": {
                "Technology & Science": 0.9157,
                "Politics": 0.8985,
                "Business": 0.7200,
                "Sports": 0.9210,
                "Entertainment": 0.8821
            },
            "business_recall": 0.7122
        }
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"    - Metadata:   {meta_path} ({os.path.getsize(meta_path)} bytes)")

    print("\n" + "=" * 70)
    print("PRODUCTION TRAINING AND ARTIFACT SERIALIZATION SUCCESSFUL")
    print("=" * 70)

    return vec_path, clf_path, pipe_path, meta_path


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv = os.path.join(base_dir, "Master_Final_Marathi_News_5Label.csv")
    models_directory = os.path.join(base_dir, "models")

    train_production_artifacts(master_csv, models_directory)

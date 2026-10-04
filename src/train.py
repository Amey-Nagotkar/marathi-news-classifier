"""
Model Training and Comparison Module
Part of Marathi News Headline Classification System (Phase 5).

Trains and objectively compares:
1. Multinomial Naive Bayes (MultinomialNB)
2. Logistic Regression (Default vs class_weight='balanced')
3. Linear Support Vector Classifier (LinearSVC: Default vs class_weight='balanced')

Fitted on data/train.csv (11,298 records) using Phase 4 TF-IDF baseline representation.
Evaluated on data/test.csv (2,825 records).
"""

import os
import sys
import time
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features import create_vectorizer, fit_vectorizer, transform_text

# Canonical label order for all evaluation and confusion matrices
LABELS_ORDER = [
    'Technology & Science',
    'Politics',
    'Business',
    'Sports',
    'Entertainment'
]


def load_data(
    train_path: str,
    test_path: str
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Loads train and test datasets from CSV using UTF-8-SIG encoding.
    Returns (X_train, y_train, X_test, y_test).
    """
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    X_train = df_train['Cleaned_Text']
    y_train = df_train['Label']
    X_test = df_test['Cleaned_Text']
    y_test = df_test['Label']

    return X_train, y_train, X_test, y_test


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns a dictionary of candidate model instances to train and compare.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear SVM": LinearSVC(
            random_state=42
        ),
        "Linear SVM (Balanced)": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),
    }


def evaluate_model(
    model: Any,
    X_train_tfidf: Any,
    y_train: pd.Series,
    X_test_tfidf: Any,
    y_test: pd.Series,
    labels: List[str] = LABELS_ORDER
) -> Dict[str, Any]:
    """
    Trains a single model, records training and prediction times,
    and calculates comprehensive evaluation metrics on the unseen test set.
    """
    # 1. Measure Training Time
    start_train = time.perf_counter()
    model.fit(X_train_tfidf, y_train)
    train_time = time.perf_counter() - start_train

    # 2. Measure Prediction Time
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test_tfidf)
    pred_time = time.perf_counter() - start_pred

    # 3. Compute Global Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average='macro', labels=labels, zero_division=0)
    macro_r = recall_score(y_test, y_pred, average='macro', labels=labels, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', labels=labels, zero_division=0)

    weighted_p = precision_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0)
    weighted_r = recall_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0)

    # 4. Per-Class Report
    clf_rep_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics = {}
    for lbl in labels:
        per_class_metrics[lbl] = {
            'precision': clf_rep_dict[lbl]['precision'],
            'recall': clf_rep_dict[lbl]['recall'],
            'f1-score': clf_rep_dict[lbl]['f1-score'],
            'support': int(clf_rep_dict[lbl]['support'])
        }

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 6. Probability support check
    supports_proba = hasattr(model, 'predict_proba')

    return {
        'model': model,
        'y_pred': y_pred,
        'train_time': train_time,
        'pred_time': pred_time,
        'accuracy': acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_p,
        'weighted_recall': weighted_r,
        'weighted_f1': weighted_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': cm,
        'supports_proba': supports_proba
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
    labels: List[str] = LABELS_ORDER
):
    """
    Renders and saves a formatted confusion matrix heatmap for a model.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[l.replace(' & ', '&\n') for l in labels],
        yticklabels=labels,
        title=f"Confusion Matrix: {model_name}",
        ylabel="True Category",
        xlabel="Predicted Category"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate values in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, format(val, 'd'),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=10, weight="bold"
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def run_model_training_pipeline(
    train_path: str,
    test_path: str,
    reports_dir: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Orchestrates the complete Phase 5 model training, evaluation, comparison,
    and report generation pipeline.
    """
    print("=" * 70)
    print("PHASE 5: MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1] Loading data:\n    Train: {train_path}\n    Test:  {test_path}")
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    print(f"    Train size: {len(X_train)} records")
    print(f"    Test size:  {len(X_test)} records")

    # 2. Extract Phase 4 Baseline TF-IDF Features
    print("\n[2] Extracting Phase 4 Baseline TF-IDF Features...")
    vectorizer = create_vectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        sublinear_tf=True,
        use_idf=True,
        lowercase=False
    )
    vectorizer, X_train_tfidf = fit_vectorizer(vectorizer, X_train)
    X_test_tfidf = transform_text(vectorizer, X_test)

    print(f"    Vocabulary Size: {len(vectorizer.vocabulary_)} features")
    print(f"    Train Matrix Shape: {X_train_tfidf.shape}")
    print(f"    Test Matrix Shape:  {X_test_tfidf.shape}")
    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1], "Feature dimension mismatch!"

    # 3. Train & Evaluate Candidate Models
    models = get_candidate_models()
    print(f"\n[3] Training and evaluating {len(models)} candidate models...")

    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    results_dict = {}
    summary_rows = []

    for name, model in models.items():
        print(f"\n  -> Training: {name} ...", end=" ", flush=True)
        eval_res = evaluate_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
        results_dict[name] = eval_res
        print(f"Done! (Train: {eval_res['train_time']:.3f}s, Pred: {eval_res['pred_time']:.4f}s)")
        print(f"     Accuracy: {eval_res['accuracy'] * 100:.2f}% | Macro F1: {eval_res['macro_f1'] * 100:.2f}% | Weighted F1: {eval_res['weighted_f1'] * 100:.2f}%")

        # Save confusion matrix plot
        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
        cm_path = os.path.join(figures_dir, safe_filename)
        save_confusion_matrix_plot(eval_res['confusion_matrix'], name, cm_path)

        summary_rows.append({
            "Model": name,
            "Accuracy": eval_res['accuracy'],
            "Macro_Precision": eval_res['macro_precision'],
            "Macro_Recall": eval_res['macro_recall'],
            "Macro_F1": eval_res['macro_f1'],
            "Weighted_Precision": eval_res['weighted_precision'],
            "Weighted_Recall": eval_res['weighted_recall'],
            "Weighted_F1": eval_res['weighted_f1'],
            "Training_Time": eval_res['train_time'],
            "Prediction_Time": eval_res['pred_time'],
            "Supports_Proba": eval_res['supports_proba']
        })

    # 4. Save Machine-Readable Results CSV
    results_df = pd.DataFrame(summary_rows)
    csv_out_path = os.path.join(reports_dir, "model_results.csv")
    results_df.to_csv(csv_out_path, index=False)
    print(f"\n[4] Saved machine-readable results to:\n    {csv_out_path}")

    # 5. Print Comparison Table
    print("\n[5] MODEL COMPARISON SUMMARY TABLE:")
    print("-" * 105)
    print(f"{'Model':<32} | {'Accuracy':<8} | {'Macro P':<8} | {'Macro R':<8} | {'Macro F1':<8} | {'Weighted F1':<11} | {'Train(s)':<8}")
    print("-" * 105)
    for r in summary_rows:
        print(f"{r['Model']:<32} | {r['Accuracy']*100:<7.2f}% | {r['Macro_Precision']*100:<7.2f}% | {r['Macro_Recall']*100:<7.2f}% | {r['Macro_F1']*100:<7.2f}% | {r['Weighted_F1']*100:<10.2f}% | {r['Training_Time']:<8.3f}")
    print("-" * 105)

    return results_df, results_dict


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "data", "train.csv")
    test_file = os.path.join(base_dir, "data", "test.csv")
    reports_directory = os.path.join(base_dir, "reports")

    run_model_training_pipeline(train_file, test_file, reports_directory)

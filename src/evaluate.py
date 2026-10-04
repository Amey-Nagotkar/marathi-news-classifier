"""
Final Model Evaluation and Error Analysis Module
Part of Marathi News Headline Classification System (Phase 6).

Evaluates the chosen production candidate:
Logistic Regression (class_weight='balanced')
fitted on Phase 4 baseline TF-IDF representation.

Generates:
1. Exact performance reproduction metrics
2. Raw and normalized confusion matrices (reports/figures/)
3. Detailed error analysis dataset (reports/error_analysis.csv)
4. Top discriminative linear features per class (reports/top_features_by_class.csv)
5. Misclassification pair rankings and confidence strata analysis
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
from src.train import LABELS_ORDER


def train_final_model(
    X_train: pd.Series,
    y_train: pd.Series
) -> Tuple[Any, LogisticRegression, Any]:
    """
    Fits baseline TF-IDF vectorizer and balanced Logistic Regression on training data.
    """
    vectorizer = create_vectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        sublinear_tf=True,
        use_idf=True,
        lowercase=False
    )
    vectorizer, X_train_tfidf = fit_vectorizer(vectorizer, X_train)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model, X_train_tfidf


def plot_confusion_matrices(
    cm: np.ndarray,
    output_raw_path: str,
    output_norm_path: str,
    labels: List[str] = LABELS_ORDER
):
    """
    Generates and saves both raw count and normalized confusion matrix plots.
    """
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)

    # 1. Raw Counts
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[l.replace(' & ', '&\n') for l in labels],
        yticklabels=labels,
        title="Final Logistic Regression (Balanced) — Confusion Matrix",
        ylabel="Actual True Category",
        xlabel="Predicted Category"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, format(val, 'd'),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, weight="bold"
            )
    fig.tight_layout()
    plt.savefig(output_raw_path)
    plt.close(fig)

    # 2. Normalized by True Class (Recall per class along diagonal)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_norm.shape[1]),
        yticks=np.arange(cm_norm.shape[0]),
        xticklabels=[l.replace(' & ', '&\n') for l in labels],
        yticklabels=labels,
        title="Final Logistic Regression (Balanced) — Normalized Confusion Matrix",
        ylabel="Actual True Category",
        xlabel="Predicted Category"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.2%}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=10, weight="bold"
            )
    fig.tight_layout()
    plt.savefig(output_norm_path)
    plt.close(fig)


def extract_error_analysis(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str]
) -> pd.DataFrame:
    """
    Identifies all misclassified test headlines and compiles top-3 predictions and probabilities.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    errors = []

    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        actual = row['Label']
        pred = y_pred[idx]

        probs = y_proba[idx]
        # Sort indices by probability descending
        sorted_indices = np.argsort(probs)[::-1]

        top_1_cls = classes[sorted_indices[0]]
        top_1_prob = probs[sorted_indices[0]]
        top_2_cls = classes[sorted_indices[1]]
        top_2_prob = probs[sorted_indices[1]]
        top_3_cls = classes[sorted_indices[2]]
        top_3_prob = probs[sorted_indices[2]]

        pred_confidence = probs[class_to_idx[pred]]

        if actual != pred:
            errors.append({
                'ID': row['ID'],
                'Text': row['Text'],
                'Cleaned_Text': row['Cleaned_Text'],
                'Actual_Label': actual,
                'Predicted_Label': pred,
                'Prediction_Confidence': round(float(pred_confidence), 4),
                'Top_1_Class': top_1_cls,
                'Top_1_Probability': round(float(top_1_prob), 4),
                'Top_2_Class': top_2_cls,
                'Top_2_Probability': round(float(top_2_prob), 4),
                'Top_3_Class': top_3_cls,
                'Top_3_Probability': round(float(top_3_prob), 4),
            })

    return pd.DataFrame(errors)


def extract_top_features(
    vectorizer: Any,
    model: LogisticRegression,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extracts top positive linear coefficients per class from Logistic Regression.
    """
    feature_names = vectorizer.get_feature_names_out()
    classes = model.classes_
    rows = []

    for class_idx, class_name in enumerate(classes):
        coefs = model.coef_[class_idx]
        top_indices = np.argsort(coefs)[::-1][:top_n]
        for rank, feat_idx in enumerate(top_indices, 1):
            rows.append({
                'Class': class_name,
                'Rank': rank,
                'Feature': feature_names[feat_idx],
                'Coefficient': round(float(coefs[feat_idx]), 4)
            })

    return pd.DataFrame(rows)


def run_full_evaluation(
    train_path: str,
    test_path: str,
    reports_dir: str
) -> Dict[str, Any]:
    """
    Executes reproduction, error analysis, confusion pair ranking,
    confidence strata analysis, feature extraction, and file generation.
    """
    print("=" * 70)
    print("PHASE 6: FINAL MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    X_train, y_train = df_train['Cleaned_Text'], df_train['Label']
    X_test, y_test = df_test['Cleaned_Text'], df_test['Label']

    # 2. Train Model
    print("\n[1] Fitting Baseline TF-IDF & Logistic Regression (Balanced)...")
    vectorizer, model, X_train_tfidf = train_final_model(X_train, y_train)
    X_test_tfidf = transform_text(vectorizer, X_test)

    # 3. Predict & Compute Probabilities
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)
    classes = list(model.classes_)

    # 4. Reproduce Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average='macro', labels=LABELS_ORDER, zero_division=0)
    macro_r = recall_score(y_test, y_pred, average='macro', labels=LABELS_ORDER, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', labels=LABELS_ORDER, zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', labels=LABELS_ORDER, zero_division=0)

    print(f"\n[2] Reproduction Verification:")
    print(f"    Accuracy:    {acc * 100:.2f}% (Expected: 89.84%)")
    print(f"    Macro F1:    {macro_f1 * 100:.2f}% (Expected: 86.75%)")
    print(f"    Weighted F1: {weighted_f1 * 100:.2f}% (Expected: 89.84%)")

    # Assert exact match with Phase 5
    assert abs(acc - 0.898407) < 1e-4, "Accuracy does not match Phase 5 results!"
    assert abs(macro_f1 - 0.867456) < 1e-4, "Macro F1 does not match Phase 5 results!"
    print("    [PASS] Exact Phase 5 results reproduced successfully.")

    # 5. Per-Class Report
    clf_rep = classification_report(y_test, y_pred, labels=LABELS_ORDER, output_dict=True, zero_division=0)
    print("\n[3] Per-Class Evaluation Breakdown:")
    print(f"    {'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'Support':<8}")
    print("    " + "-" * 70)
    for lbl in LABELS_ORDER:
        m = clf_rep[lbl]
        print(f"    {lbl:<25} | {m['precision']*100:<9.2f}% | {m['recall']*100:<9.2f}% | {m['f1-score']*100:<9.2f}% | {m['support']:<8}")

    # 6. Confusion Matrices
    cm = confusion_matrix(y_test, y_pred, labels=LABELS_ORDER)
    fig_dir = os.path.join(reports_dir, "figures")
    raw_cm_path = os.path.join(fig_dir, "final_logistic_regression_confusion_matrix.png")
    norm_cm_path = os.path.join(fig_dir, "final_logistic_regression_confusion_matrix_normalized.png")
    plot_confusion_matrices(cm, raw_cm_path, norm_cm_path, labels=LABELS_ORDER)
    print(f"\n[4] Saved Confusion Matrix plots to:\n    {raw_cm_path}\n    {norm_cm_path}")

    # 7. Error Analysis
    df_errors = extract_error_analysis(df_test, y_pred, y_proba, classes)
    error_csv_path = os.path.join(reports_dir, "error_analysis.csv")
    df_errors.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[5] Saved Error Analysis dataset ({len(df_errors)} errors) to:\n    {error_csv_path}")

    # 8. Misclassification Pairs Ranking
    confusion_pairs = df_errors.groupby(['Actual_Label', 'Predicted_Label']).size().reset_index(name='Error_Count')
    class_support = {lbl: clf_rep[lbl]['support'] for lbl in LABELS_ORDER}
    confusion_pairs['Pct_of_Actual_Class'] = confusion_pairs.apply(
        lambda r: round(r['Error_Count'] / class_support[r['Actual_Label']] * 100, 2), axis=1
    )
    confusion_pairs = confusion_pairs.sort_values(by='Error_Count', ascending=False)
    print(f"\n[6] Top 10 Misclassification Pairs:")
    print(confusion_pairs.head(10).to_string(index=False))

    # 9. Confidence Strata Breakdown
    high_conf_errors = df_errors[df_errors['Prediction_Confidence'] >= 0.70]
    med_conf_errors = df_errors[(df_errors['Prediction_Confidence'] >= 0.40) & (df_errors['Prediction_Confidence'] < 0.70)]
    low_conf_errors = df_errors[df_errors['Prediction_Confidence'] < 0.40]

    print(f"\n[7] Error Confidence Breakdown (Total Errors: {len(df_errors)}):")
    print(f"    High Model Probability (>= 70%): {len(high_conf_errors)} ({len(high_conf_errors)/len(df_errors)*100:.2f}%)")
    print(f"    Medium Model Probability (40-69%): {len(med_conf_errors)} ({len(med_conf_errors)/len(df_errors)*100:.2f}%)")
    print(f"    Low Model Probability (< 40%):    {len(low_conf_errors)} ({len(low_conf_errors)/len(df_errors)*100:.2f}%)")

    # 10. Extract Top Features per Class
    df_top_feats = extract_top_features(vectorizer, model, top_n=20)
    top_feats_csv_path = os.path.join(reports_dir, "top_features_by_class.csv")
    df_top_feats.to_csv(top_feats_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[8] Saved Top Features by Class to:\n    {top_feats_csv_path}")

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'clf_rep': clf_rep,
        'cm': cm,
        'df_errors': df_errors,
        'confusion_pairs': confusion_pairs,
        'high_conf_errors': high_conf_errors,
        'med_conf_errors': med_conf_errors,
        'low_conf_errors': low_conf_errors,
        'df_top_feats': df_top_feats
    }


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "data", "train.csv")
    test_file = os.path.join(base_dir, "data", "test.csv")
    reports_directory = os.path.join(base_dir, "reports")

    run_full_evaluation(train_file, test_file, reports_directory)

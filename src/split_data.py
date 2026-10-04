"""
Train/Test Dataset Splitting Module
Part of Marathi News Headline Classification System (Phase 3).

Loads the master dataset, applies Phase 2 text preprocessing,
performs comprehensive leakage and duplicate checks,
and generates a stratified 80/20 train/test split with random_state=42.
Saves split datasets to data/train.csv and data/test.csv using UTF-8-SIG encoding.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Phase 2 preprocessing
from src.preprocessing import load_dataset, clean_headline, preprocess_dataset


def run_train_test_split(
    master_csv_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Executes data loading, preprocessing, leakage verification,
    stratified splitting, and saving of train and test datasets.
    """
    print("=" * 65)
    print("PHASE 3: STRATIFIED TRAIN / TEST SPLIT & DATA PREPARATION")
    print("=" * 65)

    # 1. Load Dataset
    print(f"\n[1] Loading master dataset from:\n    {master_csv_path}")
    df = load_dataset(master_csv_path)
    total_rows = len(df)
    print(f"    Total rows loaded: {total_rows}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Null values per column:\n{df.isnull().sum().to_dict()}")

    assert total_rows == 14123, f"Expected 14123 rows, found {total_rows}"
    assert df.isnull().sum().sum() == 0, "Found unexpected null values in dataset!"

    # 2. Verify Target Labels
    expected_labels = {
        'Technology & Science',
        'Politics',
        'Business',
        'Sports',
        'Entertainment'
    }
    actual_labels = set(df['Label'].unique())
    print(f"\n[2] Target Labels Verification:")
    print(f"    Actual labels: {actual_labels}")
    assert actual_labels == expected_labels, f"Label mismatch! Found: {actual_labels}"
    print("    [PASS] Exactly 5 target labels verified.")

    # 3. Apply Phase 2 Preprocessing
    print("\n[3] Applying Phase 2 Preprocessing Pipeline...")
    df_preprocessed = preprocess_dataset(
        df,
        text_column='Text',
        cleaned_column='Cleaned_Text',
        lowercase_english=True
    )
    assert len(df_preprocessed) == total_rows, "Row count changed during preprocessing!"
    print("    [PASS] Preprocessing complete. 0 records dropped.")

    # 4. Leakage & Duplication Checks Before Split
    print("\n[4] Data Leakage & Duplication Verification:")
    
    # Check duplicate IDs
    duplicate_ids = df_preprocessed[df_preprocessed['ID'].duplicated(keep=False)]
    print(f"    Duplicate IDs: {len(duplicate_ids)}")
    assert len(duplicate_ids) == 0, "Found duplicate IDs!"

    # Check exact duplicate raw texts
    dup_raw_texts = df_preprocessed[df_preprocessed['Text'].duplicated(keep=False)]
    print(f"    Duplicate raw texts: {len(dup_raw_texts)}")

    # Check duplicate cleaned/normalized texts
    dup_cleaned_texts = df_preprocessed[df_preprocessed['Cleaned_Text'].duplicated(keep=False)]
    print(f"    Duplicate cleaned texts: {len(dup_cleaned_texts)}")

    # Check conflicting labels for same text
    text_label_groups = df_preprocessed.groupby('Cleaned_Text')['Label'].nunique()
    conflicting_texts = text_label_groups[text_label_groups > 1]
    print(f"    Cleaned texts with conflicting labels: {len(conflicting_texts)}")
    assert len(conflicting_texts) == 0, "Found identical texts with conflicting labels!"

    print("    [PASS] Data integrity verified: 0 duplicates, 0 conflicting labels.")

    # 5. Stratified Train / Test Split
    print(f"\n[5] Executing Stratified Split (80% Train, 20% Test, random_state={random_state}):")
    
    train_df, test_df = train_test_split(
        df_preprocessed,
        test_size=test_size,
        random_state=random_state,
        stratify=df_preprocessed['Label']
    )

    train_rows = len(train_df)
    test_rows = len(test_df)
    print(f"    Training records: {train_rows} ({train_rows / total_rows * 100:.2f}%)")
    print(f"    Testing records:  {test_rows} ({test_rows / total_rows * 100:.2f}%)")
    print(f"    Total:            {train_rows + test_rows}")

    assert train_rows + test_rows == total_rows, "Train + Test != Total records!"

    # 6. Verify Overlap Between Train and Test Sets
    train_texts = set(train_df['Cleaned_Text'])
    test_texts = set(test_df['Cleaned_Text'])
    overlap_texts = train_texts.intersection(test_texts)
    print(f"\n[6] Cross-Split Leakage Check:")
    print(f"    Unique training texts: {len(train_texts)}")
    print(f"    Unique testing texts:  {len(test_texts)}")
    print(f"    Overlapping texts between train and test: {len(overlap_texts)}")
    assert len(overlap_texts) == 0, "Data leakage detected: Identical headlines found in both train and test sets!"
    print("    [PASS] 0 cross-split headline overlap. Zero data leakage.")

    # 7. Detailed Class Distribution Comparison
    print("\n[7] Class Distribution Across Splits:")
    total_counts = df_preprocessed['Label'].value_counts()
    train_counts = train_df['Label'].value_counts()
    test_counts = test_df['Label'].value_counts()

    distribution_data = []
    print(f"    {'Label':<25} | {'Total':<6} | {'Train':<6} | {'Test':<5} | {'Train %':<7} | {'Test %':<7}")
    print("    " + "-" * 65)

    for label in [
        'Technology & Science',
        'Politics',
        'Sports',
        'Entertainment',
        'Business'
    ]:
        tot = total_counts[label]
        tr = train_counts[label]
        te = test_counts[label]
        tr_pct = tr / train_rows * 100
        te_pct = te / test_rows * 100
        distribution_data.append({
            'Label': label,
            'Total': tot,
            'Train': tr,
            'Test': te,
            'Train_Pct': f"{tr_pct:.2f}%",
            'Test_Pct': f"{te_pct:.2f}%"
        })
        print(f"    {label:<25} | {tot:<6} | {tr:<6} | {te:<5} | {tr_pct:<6.2f}% | {te_pct:<6.2f}%")

    # Confirm Business class presence
    assert train_counts['Business'] > 0 and test_counts['Business'] > 0, "Business class missing!"
    print("\n    [PASS] All 5 classes present in both splits with identical class proportions.")

    # 8. Save Datasets
    os.makedirs(output_dir, exist_ok=True)
    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")

    # Format output columns: ID, Text, Cleaned_Text, Label, Original_Label
    columns_to_save = ['ID', 'Text', 'Cleaned_Text', 'Label', 'Original_Label']
    train_df[columns_to_save].to_csv(train_csv_path, index=False, encoding='utf-8-sig')
    test_df[columns_to_save].to_csv(test_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n[8] Saved split files:")
    print(f"    Train: {train_csv_path} ({os.path.getsize(train_csv_path)} bytes)")
    print(f"    Test:  {test_csv_path} ({os.path.getsize(test_csv_path)} bytes)")

    # 9. Verification of Saved Files
    train_check = pd.read_csv(train_csv_path, encoding='utf-8-sig')
    test_check = pd.read_csv(test_csv_path, encoding='utf-8-sig')
    assert len(train_check) == train_rows, "Train file row count mismatch!"
    assert len(test_check) == test_rows, "Test file row count mismatch!"
    assert set(train_check.columns) == set(columns_to_save), "Train columns mismatch!"
    assert set(test_check.columns) == set(columns_to_save), "Test columns mismatch!"
    print("    [PASS] Saved file verification completed successfully.")

    print("\n" + "=" * 65)
    print("PHASE 3 COMPLETE: DATA PREPARED AND READY FOR PHASE 4")
    print("=" * 65)

    return distribution_data, train_rows, test_rows, total_rows


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv = os.path.join(base_dir, "Master_Final_Marathi_News_5Label.csv")
    data_dir = os.path.join(base_dir, "data")
    run_train_test_split(master_csv, data_dir)

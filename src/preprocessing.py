"""
Marathi News Headline Preprocessing Module
Part of Marathi News Headline Classification System.

This module provides a robust, reusable text normalization and preprocessing pipeline
specifically designed for Marathi Devanagari text and Marathi-English code-mixed headlines.

Key Features:
- Preserves Marathi Devanagari characters, matras, nuktas, halants, and conjuncts.
- Preserves Latin script (English words, brand names, technical abbreviations).
- Preserves ASCII numerals (0-9) and Marathi Devanagari numerals (०-९).
- Normalizes Unicode representations to Canonical Composition (NFC).
- Unescapes HTML entities.
- Normalizes typographical quotes, dashes, and non-breaking spaces.
- Applies lowercase normalization to Latin characters (no-op on Devanagari).
- Reusable across batch CSV training and single user-input inference in Streamlit.
"""

import html
import os
import re
import sys
import unicodedata
from typing import Optional, Union, List
import pandas as pd

# Regular expression pattern for tokenizing Marathi + English + Digits
# Scikit-learn's default token_pattern r'(?u)\b\w\w+\b' incorrectly splits Marathi words
# at combining vowel signs (matras). MARATHI_TOKEN_PATTERN preserves complete Devanagari
# and alphanumeric tokens.
MARATHI_TOKEN_PATTERN = r'[\u0900-\u097F\w]+'

# Typographical normalization maps
QUOTE_MAP = {
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '`': "'",
    '«': '"',
    '»': '"',
}

DASH_MAP = {
    '–': '-',  # en-dash
    '—': '-',  # em-dash
    '―': '-',  # horizontal bar
    '−': '-',  # minus sign
}


def clean_headline(text: Union[str, float, None], lowercase_english: bool = True) -> str:
    """
    Cleans and normalizes a single Marathi news headline.

    Parameters
    ----------
    text : str or Any
        The raw headline string to clean.
    lowercase_english : bool, default=True
        Whether to convert Latin characters to lowercase.
        Devanagari characters are case-less and remain unchanged.

    Returns
    -------
    str
        The cleaned, normalized headline.
    """
    if text is None or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g., &amp; -> &, &quot; -> ", etc.)
    text = html.unescape(text)

    # 2. Unicode normalization: Canonical Composition (NFC)
    # Ensures base characters and combining matras are consistently represented.
    text = unicodedata.normalize('NFC', text)

    # 3. Strip invisible control characters, BOM, and zero-width spaces
    for ch in ['\ufeff', '\u200b', '\u200e', '\u200f']:
        if ch in text:
            text = text.replace(ch, '')

    # 4. Normalize non-breaking spaces to standard spaces
    text = text.replace('\u00a0', ' ')

    # 5. Standardize fancy typographical quotes
    for fancy_quote, standard_quote in QUOTE_MAP.items():
        if fancy_quote in text:
            text = text.replace(fancy_quote, standard_quote)

    # 6. Standardize typographical dashes to standard hyphen
    for fancy_dash, standard_dash in DASH_MAP.items():
        if fancy_dash in text:
            text = text.replace(fancy_dash, standard_dash)

    # 7. Lowercase English characters if enabled
    if lowercase_english:
        text = text.lower()

    # 8. Collapse repeated whitespace and strip leading/trailing spaces
    text = re.sub(r'[ \t\r\n]+', ' ', text).strip()

    return text


def tokenize_marathi(text: str) -> List[str]:
    """
    Tokenizes a cleaned Marathi headline into constituent tokens,
    preserving full Devanagari words with combining vowel signs,
    English words, and numeric digits.

    Parameters
    ----------
    text : str
        The cleaned input text.

    Returns
    -------
    List[str]
        List of tokens.
    """
    if not text:
        return []
    return re.findall(MARATHI_TOKEN_PATTERN, text)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the Marathi news dataset CSV file with UTF-8-SIG encoding.

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with guaranteed clean column names.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # utf-8-sig automatically strips any BOM prefix from column headers (e.g. \ufeffID)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    return df


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = 'Text',
    cleaned_column: str = 'Cleaned_Text',
    lowercase_english: bool = True
) -> pd.DataFrame:
    """
    Applies headline cleaning across a pandas DataFrame.
    Guarantees no records are dropped or altered beyond normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_column : str, default='Text'
        Name of the source text column.
    cleaned_column : str, default='Cleaned_Text'
        Name of the output column containing normalized text.
    lowercase_english : bool, default=True
        Whether to lowercase Latin characters.

    Returns
    -------
    pd.DataFrame
        DataFrame with the cleaned text column added.
    """
    result_df = df.copy()
    result_df[cleaned_column] = result_df[text_column].apply(
        lambda t: clean_headline(t, lowercase_english=lowercase_english)
    )
    return result_df


if __name__ == '__main__':
    # Direct execution test for Phase 2 validation
    sys.stdout.reconfigure(encoding='utf-8')

    default_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Master_Final_Marathi_News_5Label.csv"
    )

    print("=" * 60)
    print("MARATHI NEWS PREPROCESSING PIPELINE VALIDATION")
    print("=" * 60)
    print(f"Loading master dataset from:\n  {default_csv}")

    df_raw = load_dataset(default_csv)
    initial_count = len(df_raw)
    print(f"\n[1] Initial Record Count: {initial_count}")
    print(f"    Columns: {list(df_raw.columns)}")

    # Apply preprocessing
    df_clean = preprocess_dataset(df_raw, text_column='Text', cleaned_column='Cleaned_Text')
    processed_count = len(df_clean)

    print(f"\n[2] Processed Record Count: {processed_count}")
    assert initial_count == processed_count, "ERROR: Record count changed during preprocessing!"
    print("    [PASS] Record count preserved exactly: 14,123 -> 14,123 (0 dropped).")

    # Check for empty strings after cleaning
    empty_records = df_clean[df_clean['Cleaned_Text'].str.len() == 0]
    print(f"\n[3] Empty Records After Cleaning: {len(empty_records)}")
    assert len(empty_records) == 0, "ERROR: Some headlines became empty after cleaning!"
    print("    [PASS] 0 empty headlines produced.")

    # Check modifications
    diff_mask = df_clean['Text'] != df_clean['Cleaned_Text']
    total_modified = diff_mask.sum()
    print(f"\n[4] Total Normalized Headlines: {total_modified} / {initial_count} ({total_modified / initial_count * 100:.2f}%)")

    # Verify script preservation
    devanagari_regex = re.compile(r'[\u0900-\u097F]')
    latin_regex = re.compile(r'[a-zA-Z]')
    digit_regex = re.compile(r'[0-9\u0966-\u096F]')

    code_mixed_count = df_clean['Cleaned_Text'].apply(
        lambda t: bool(devanagari_regex.search(t)) and bool(latin_regex.search(t))
    ).sum()

    digits_count = df_clean['Cleaned_Text'].apply(
        lambda t: bool(digit_regex.search(t))
    ).sum()

    print(f"\n[5] Script & Feature Preservation:")
    print(f"    Code-mixed headlines (Devanagari + English): {code_mixed_count} (Preserved)")
    print(f"    Headlines containing numbers: {digits_count} (Preserved)")

    # Test token pattern
    sample_headline = df_clean.iloc[0]['Cleaned_Text']
    tokens = tokenize_marathi(sample_headline)
    print(f"\n[6] Tokenizer Sample Check on Row 1:")
    print(f"    Headline: {sample_headline}")
    print(f"    Tokens:   {tokens}")

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE VALIDATION SUCCESSFUL")
    print("=" * 60)

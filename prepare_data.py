
# ============================================================
# Phase 2 - Data Preparation
# Project: AI-Based Phishing Detection System for Cameroon
# Author: TCHATCHOUA NGASSAM TRESOR LARRY
# Description: Load, clean and prepare the SMS dataset
# ============================================================

import pandas as pd
import os

# ── Step 1: Load the UCI SMS Spam Collection dataset ──
print("Loading UCI SMS Spam Collection dataset...")

uci_path = "dataset/SMSSpamCollection"

uci_data = pd.read_csv(
    uci_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print(f"Dataset loaded successfully!")
print(f"Total messages: {len(uci_data)}")
print(f"Sample data:")
print(uci_data.head())

# ── Step 2: Check label distribution ──
print("\nLabel distribution:")
print(uci_data["label"].value_counts())

# ── Step 3: Convert ham/spam to numbers ──
print("\nConverting labels...")

def convert_label(label):
    if label == "ham":
        return 0   # Legitimate
    elif label == "spam":
        return 2   # Phishing
    else:
        return 1   # Suspicious

uci_data["label_num"] = uci_data["label"].apply(convert_label)

# ── Step 4: Add source column ──
uci_data["source"] = "SMS"

# ── Step 5: Keep only useful columns ──
uci_clean = uci_data[["message", "source", "label_num"]].copy()
uci_clean.columns = ["message", "source", "label"]

# ── Step 6: Save cleaned dataset ──
output_path = "dataset/uci_clean.csv"
uci_clean.to_csv(output_path, index=False, encoding="utf-8")

print(f"\nCleaned dataset saved to: {output_path}")
print(f"Total messages saved: {len(uci_clean)}")
print(f"\nLabel breakdown:")
print(uci_clean["label"].value_counts())
print("\nSample of cleaned data:")
print(uci_clean.head(10))
print("\nDone! UCI dataset is ready.")
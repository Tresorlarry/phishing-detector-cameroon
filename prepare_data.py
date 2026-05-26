
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
email_data = pd.read_csv("dataset/Phishing_Email.csv")
print("Email dataset columns: " + str(email_data.columns.tolist()))
print("Total emails: " + str(len(email_data)))
print(email_data.head())
email_data = pd.read_csv("dataset/Phishing_Email.csv")

email_data = email_data[["Email Text", "Email Type"]].copy()
email_data.columns = ["message", "label"]

email_data["label"] = email_data["label"].map({
    "Safe Email": 0,
    "Phishing Email": 2
})

email_data["source"] = "Email"
email_data = email_data.dropna()

email_clean = email_data[["message", "source", "label"]].copy()
email_clean.to_csv("dataset/email_clean.csv", index=False)

print("Email dataset cleaned!")
print("Total emails: " + str(len(email_clean)))
print(email_clean["label"].value_counts())
cameroon = pd.read_csv("dataset/cameroon_dataset.csv")
uci = pd.read_csv("dataset/uci_clean.csv")
email = pd.read_csv("dataset/email_clean.csv")

final = pd.concat([uci, email, cameroon], ignore_index=True)
final = final.dropna()

final.to_csv("dataset/final_dataset.csv", index=False)

print("Final dataset created!")
print("Total messages: " + str(len(final)))
print(final["label"].value_counts())

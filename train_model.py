import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import string

print("Loading dataset...")
df = pd.read_csv("dataset/final_dataset.csv")
df = df.dropna(subset=["message", "label"])
print("Total messages: " + str(len(df)))
print(df["label"].value_counts())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaning text...")
df["clean_message"] = df["message"].apply(clean_text)
print("Sample cleaned message:")
print(df["clean_message"][0])

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set: " + str(len(X_train)))
print("Testing set: " + str(len(X_test)))

print("\nExtracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Features extracted: " + str(X_train_tfidf.shape[1]))

print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy: " + str(round(nb_acc * 100, 2)) + "%")

print("\nTraining SVM...")
svm_model = LinearSVC(max_iter=2000)
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, svm_pred)
print("SVM Accuracy: " + str(round(svm_acc * 100, 2)) + "%")

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy: " + str(round(rf_acc * 100, 2)) + "%")

print("\n========== MODEL COMPARISON ==========")
print("Naive Bayes:   " + str(round(nb_acc * 100, 2)) + "%")
print("SVM:           " + str(round(svm_acc * 100, 2)) + "%")
print("Random Forest: " + str(round(rf_acc * 100, 2)) + "%")

accuracies = {"Naive Bayes": nb_acc, "SVM": svm_acc, "Random Forest": rf_acc}
best_name = max(accuracies, key=accuracies.get)
best_model = {"Naive Bayes": nb_model, "SVM": svm_model, "Random Forest": rf_model}[best_name]
print("\nBest model: " + best_name)

print("\nDetailed report for best model:")
best_pred = {"Naive Bayes": nb_pred, "SVM": svm_pred, "Random Forest": rf_pred}[best_name]
print(classification_report(y_test, best_pred, target_names=["Legitimate","Suspicious","Phishing"]))

print("\nSaving best model and vectorizer...")
joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
joblib.dump(best_name, "model/best_model_name.pkl")
print("Model saved to model/best_model.pkl")
print("Vectorizer saved to model/tfidf_vectorizer.pkl")
print("\nPhase 3 Complete!")
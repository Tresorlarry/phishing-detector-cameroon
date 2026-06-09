import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize

os.makedirs("static/graphs", exist_ok=True)

print("="*55)
print("  PHISHING DETECTION - AI MODEL TRAINING")
print("  ICT University Cameroon - Tresor Larry")
print("="*55)

print("\n[1/7] Loading dataset...")
df = pd.read_csv("dataset/final_dataset.csv")
df = df.dropna(subset=["message", "label"])
df["label"] = df["label"].astype(int)
print("Total messages: " + str(len(df)))
print(df["label"].value_counts())

print("\n[2/7] Cleaning text...")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)
print("Text cleaning done!")

print("\n[3/7] Generating dataset graphs...")

plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c', '#f39c12']
label_names = {0: 'Legitimate', 1: 'Suspicious', 2: 'Phishing'}
counts = df["label"].value_counts().sort_index()
bars = plt.bar([label_names[i] for i in counts.index],
               counts.values, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.title('Dataset Label Distribution', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Message Class', fontsize=13)
plt.ylabel('Number of Messages', fontsize=13)
plt.tight_layout()
plt.savefig('static/graphs/label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Label distribution chart saved!")

plt.figure(figsize=(7, 7))
pie_colors = ['#2ecc71', '#f39c12', '#e74c3c']
wedges, texts, autotexts = plt.pie(
    counts.values,
    labels=[label_names[i] for i in counts.index],
    autopct='%1.1f%%',
    colors=pie_colors,
    startangle=90,
    explode=[0.05]*len(counts),
    shadow=True
)
for text in autotexts:
    text.set_fontsize(13)
    text.set_fontweight('bold')
plt.title('Dataset Composition', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/graphs/label_pie_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Pie chart saved!")

print("\n[4/7] Preparing features...")
X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training set: " + str(len(X_train)))
print("Testing set:  " + str(len(X_test)))
print("Features:     5000 TF-IDF")

print("\n[5/7] Training 6 models...")
models = {
    "Naive Bayes":         MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 LinearSVC(max_iter=2000, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print("  Training " + name + "...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results[name] = {"model": model, "accuracy": acc,
                     "precision": prec, "recall": rec, "f1": f1, "predictions": y_pred}
    print("    Accuracy: " + str(round(acc*100, 2)) + "%")

print("\n[6/7] Generating model graphs...")

model_names = list(results.keys())
accuracies  = [results[m]["accuracy"]*100 for m in model_names]
precisions  = [results[m]["precision"]*100 for m in model_names]
recalls     = [results[m]["recall"]*100 for m in model_names]
f1_scores   = [results[m]["f1"]*100 for m in model_names]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.2
bars1 = ax.bar(x - 1.5*width, accuracies,  width, label='Accuracy',  color='#3498db', edgecolor='white')
bars2 = ax.bar(x - 0.5*width, precisions,  width, label='Precision', color='#2ecc71', edgecolor='white')
bars3 = ax.bar(x + 0.5*width, recalls,     width, label='Recall',    color='#e74c3c', edgecolor='white')
bars4 = ax.bar(x + 1.5*width, f1_scores,   width, label='F1-Score',  color='#f39c12', edgecolor='white')
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                str(round(h, 1)) + '%', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.set_xlabel('Model', fontsize=13)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_title('Model Performance Comparison — All 6 Models', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/graphs/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model comparison chart saved!")

best_name = max(results, key=lambda m: results[m]["accuracy"])
best_pred = results[best_name]["predictions"]
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legitimate','Suspicious','Phishing'],
            yticklabels=['Legitimate','Suspicious','Phishing'],
            linewidths=0.5, linecolor='white', annot_kws={"size": 13})
plt.title('Confusion Matrix — ' + best_name, fontsize=15, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig('static/graphs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved!")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, (name, color) in enumerate([('Naive Bayes','#3498db'), ('SVM','#e74c3c')]):
    vals = [results[name]['accuracy']*100, results[name]['precision']*100,
            results[name]['recall']*100, results[name]['f1']*100]
    bars = axes[idx].bar(metrics, vals, color=color, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                       str(round(val, 1)) + '%', ha='center', va='bottom',
                       fontweight='bold', fontsize=11)
    axes[idx].set_title(name + ' — Detailed Metrics', fontsize=13, fontweight='bold')
    axes[idx].set_ylim(0, 110)
    axes[idx].set_ylabel('Score (%)', fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)
plt.suptitle('Detailed Model Metrics Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('static/graphs/detailed_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Detailed metrics chart saved!")

feat_names = tfidf.get_feature_names_out()
nb_model = results["Naive Bayes"]["model"]
log_probs = nb_model.feature_log_prob_
phishing_idx = list(nb_model.classes_).index(2) if 2 in nb_model.classes_ else -1
if phishing_idx >= 0:
    top_indices = log_probs[phishing_idx].argsort()[-20:][::-1]
    top_features = [feat_names[i] for i in top_indices]
    top_scores   = [log_probs[phishing_idx][i] for i in top_indices]
    plt.figure(figsize=(10, 7))
    colors_feat = ['#e74c3c' if s > np.median(top_scores) else '#f39c12' for s in top_scores]
    bars = plt.barh(top_features[::-1], [abs(s) for s in top_scores[::-1]],
                    color=colors_feat[::-1], edgecolor='white')
    plt.title('Top 20 Phishing Keywords — Feature Importance', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Feature Weight (Log Probability)', fontsize=13)
    plt.ylabel('Keywords & Phrases', fontsize=13)
    plt.tight_layout()
    plt.savefig('static/graphs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance chart saved!")

print("\n[7/7] Saving best model...")
best_model = results[best_name]["model"]
joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
joblib.dump(best_name, "model/best_model_name.pkl")
joblib.dump(results, "model/all_results.pkl")

print("\n" + "="*55)
print("  MODEL COMPARISON SUMMARY")
print("="*55)
for name in model_names:
    marker = " <-- BEST" if name == best_name else ""
    print("  {:<22} {:.2f}%{}".format(name, results[name]['accuracy']*100, marker))
print("="*55)
print("\nDetailed report for best model (" + best_name + "):")
print(classification_report(y_test, best_pred,
      target_names=['Legitimate','Suspicious','Phishing'], zero_division=0))
print("\nAll graphs saved to: static/graphs/")
print("Best model saved to: model/best_model.pkl")
print("\nPhase 3 COMPLETE!")
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "saved_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

dataset = load_dataset("ag_news")
test_data = dataset["test"]

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=False
)

preds = []
texts = []
labels = []

for item in test_data:
    texts.append(item["text"])
    labels.append(item["label"])
    pred = classifier(item["text"], truncation=True)[0]["label"]
    preds.append(int(pred.split("_")[-1]))

# error analysis ----------
print("Classification Report:\n")
print(classification_report(labels, preds))

print("Confusion Matrix:\n")
print(confusion_matrix(labels, preds))

# qualitative error sampling ----------
errors = []
for text, y_true, y_pred in zip(texts, labels, preds):
    if y_true != y_pred:
        errors.append({
            "text": text,
            "true_label": y_true,
            "pred_label": y_pred
        })

error_df = pd.DataFrame(errors)
print("Sample Misclassified Cases:\n")
print(error_df.head(5))

# robustness test
def perturb_text(text):
    words = text.split()
    if len(words) > 6:
        words.pop(len(words)//2)
    return " ".join(words)

robustness_samples = texts[:50]

print("\nRobustness Test:\n")
for txt in robustness_samples[:5]:
    original = classifier(txt)[0]
    perturbed = classifier(perturb_text(txt))[0]
    print("Original:", original)
    print("Perturbed:", perturbed)
    print("---")


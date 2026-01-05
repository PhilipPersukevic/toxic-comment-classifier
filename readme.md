# Toxic Comment Classifier

## Overview
This project is a **machine learning classifier** for detecting comments containing **identity-based hate** (`identity_hate`).  
The classifier uses **Logistic Regression** with **TF-IDF features** and **custom tokenization**, including punctuation removal, stopword removal, and stemming.

The dataset is highly **imbalanced**, with far fewer `identity_hate` comments compared to non-hateful comments, so the pipeline uses strategies to handle this imbalance.

---

## Features
- **Custom tokenization**: tokenizes text, removes punctuation and stopwords, applies stemming.
- **TF-IDF vectorization** of text comments.
- **Logistic Regression** with `class_weight="balanced"` to handle class imbalance.
- **Precision-Recall curve analysis** for choosing the optimal threshold.
- **Prediction function** for new comments.

---

## Dataset
The dataset files include:
- `train.csv` – training data with `comment_text` and `identity_hate` labels.
- `test.csv` – test data for evaluation.
- `sample_submission.csv` and `test.labels.csv` – optional for competitions or testing.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/toxic-comment-classifier.git
cd toxic-comment-classifier

2. (Optional) Create a virtual environment:

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Download NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

Usage
Train the model
from main import model_pipeline, train_df

# Train the pipeline
model_pipeline.fit(train_df["comment_text"], train_df["identity_hate"])

Predict new comments

from main import predict_comment

comment = "You are such an idiot!"
label, prob = predict_comment(comment)
print("Label:", label, "Probability:", prob)

Label = 1 → comment contains identity_hate

Label = 0 → comment is not hateful

Probability → model's confidence score

Evaluate on test set

from main import model_pipeline, test_df
from sklearn.metrics import precision_score, recall_score
import numpy as np

y_true = test_df["identity_hate"]
y_proba = model_pipeline.predict_proba(test_df["comment_text"])[:,1]

# Use chosen threshold for high precision
best_threshold = 0.9996
y_pred_thresh = (y_proba >= best_threshold).astype(int)

precision = precision_score(y_true, y_pred_thresh)
recall = recall_score(y_true, y_pred_thresh)

print("Precision:", precision)
print("Recall:", recall)

Example Output
Default threshold (0.5):

Precision: 0.14
Recall: 0.86

Threshold adjusted for high precision (≈0.9996):
Precision: 0.83
Recall: 0.018

Example predictions:
Comment: Hi, my name is Filip
Predicted label: 0, Probability: 0.118

Comment: You are such an idiot! I can't stand people like you.
Predicted label: 1, Probability: 0.772

Comment: I love this community!
Predicted label: 0, Probability: 0.365

High precision threshold ensures most predicted identity_hate comments are correct.
Trade-off: low recall because dataset is highly imbalanced.

Model
The trained pipeline is saved as identity_hate_model.pkl and can be loaded using joblib:
import joblib

model_pipeline = joblib.load("identity_hate_model.pkl")

Now you can use predict_comment() with the trained model.
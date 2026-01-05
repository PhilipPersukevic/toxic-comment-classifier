# Toxic Comment Classifier
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay
from matplotlib import pyplot as plt
import numpy as np
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/train.csv", sep=",")
print("Dataset shape:", df.shape)
print(df.head(5))
print("Class distribution:\n", df["identity_hate"].value_counts())

# Show few examples of positive (identity_hate=1) and negative comments
print("\nSample positive comments:")
for c in df[df["identity_hate"] == 1]["comment_text"].head(5):
    print(c)
    print('---------------------')
for c in df[df["identity_hate"] == 0]["comment_text"].head(5):
    print(c)
    print('=====================')

# Split data into train/test
train_df, test_df = train_test_split(
    df, 
    test_size=0.2,
    random_state=42,
    stratify=df["identity_hate"]
)

# Handle class imbalance by undersampling the negative class
df_pros = train_df[train_df["identity_hate"] == 1]
df_neg = train_df[train_df["identity_hate"] == 0].sample(5 * len(df_pros), random_state = 42)

train_df = pd.concat([df_pros, df_neg]).sample(frac = 1, random_state = 42)

print("\nTrain\Test shapes:", train_df.shape, test_df.shape)
print("Train class distribution:\n", train_df["identity_hate"].value_counts())
print("Test class distribution:\n", test_df["identity_hate"].value_counts())

# Example: tokenization and processing
sentence_example = df.iloc[1]["comment_text"]

# Tokenize and clean
tokens = word_tokenize(sentence_example)
tokens_without_punctuation = [token for token in tokens if token not in string.punctuation]
english_stop_words = set(stopwords.words("english"))
tokens_without_stop_words_and_punctuation = [token for token in tokens_without_punctuation if token.lower() not in english_stop_words]

# Apply stemming
snowball = SnowballStemmer(language="english")
stemmed_tokens = [snowball.stem(token) for token in tokens_without_stop_words_and_punctuation]

print("\nExample preprocessing:")
print("Original:", sentence_example)
print("Tokens:", tokens)
print("Tokens without punctuation:", tokens_without_punctuation)
print("Tokens without stopwords:", tokens_without_stop_words_and_punctuation)
print("Stemmed tokens:", stemmed_tokens)

# Define reusable tokenization function
def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence)
    tokens = [token for token in tokens if token not in string.punctuation]
    if remove_stop_words:
        tokens = [token for token in tokens if token.lower() not in english_stop_words]
    tokens = [snowball.stem(token) for token in tokens]
    return tokens

# Function to use in TF-IDF vectorizer (cannot use lambda because of joblib pickle issue)
def tokenizer_for_vectorizer(text):
    return tokenize_sentence(text, remove_stop_words=True)

# Create TF-IDF + Logistic Regression pipeline
model_pipeline = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            tokenizer=tokenizer_for_vectorizer,
            lowercase=False,
            token_pattern=None
        )
    ),
    (
        "model",
        LogisticRegression(
            random_state=0,
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

# Train the model
model_pipeline.fit(train_df["comment_text"], train_df["identity_hate"])

# Test on individual comments
example_comments = [
    "Hi, my name is Filip",
    "You are such an idiot! I can't stand people like you.",
    "I love this community!"
]

print("\nExample predictions:")
for comment in example_comments:
    prob = model_pipeline.predict_proba([comment])[0,1]
    label = 1 if prob >= 0.5 else 0
    print(f"Comment: {comment}\nPredicted label: {label}, Probability: {prob:.3f}\n")

# Evaluate on test set
y_true = test_df["identity_hate"]
y_proba = model_pipeline.predict_proba(test_df["comment_text"])[:,1]
y_pred = (y_proba >= 0.5).astype(int)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print("Test set metrics eith threshold 0.5:")
print("Precision:", precision)
print("Recall:", recall)

# Precision-Recall curve and threshold selection
prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
disp = PrecisionRecallDisplay(precision=prec, recall=rec)
disp.plot()
plt.title("Precision-Recall Curve")
plt.show()

# Choose threshold where precision >= 0.8
best_idx = np.argmax(prec >= 0.8)
best_threshold = thresholds[best_idx]
print("Selected threshold for precision >= 0.8:", best_threshold)

# Metrics with chosen threshold
y_pred_thresh = (y_proba >= best_threshold).astype(int)
precision_at_thresh = precision_score(y_true, y_pred_thresh)
recall_at_thresh = recall_score(y_true, y_pred_thresh)
print("Metrics at chosen threshold:")
print("Precision:", precision_at_thresh)
print("Recall:", recall_at_thresh)

# Function to predict new comments using chosen threshold
def predict_comment(comment, threshold=best_threshold):
    prob = model_pipeline.predict_proba([comment])[0,1]
    label = 1 if prob >= threshold else 0
    return label, prob

# Example usage
label, prob = predict_comment("You are stupid!")
print("\nPrediction example:")
print("Comment: 'You are stupid!'")
print("Label:", label, "Probability:", prob)

# Save trained pipeline to file
joblib.dump(model_pipeline, "identity_hate_model.pkl")
print("\nModel saved as 'identity_hate_model.pkl'")
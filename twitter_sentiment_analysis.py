"""
=============================================================
  Twitter Sentiment Analysis — NLP + Machine Learning
=============================================================
  Pipeline:
    1. Data loading (sample built-in or Sentiment140 CSV)
    2. Text preprocessing (clean, tokenize, lemmatize)
    3. Feature extraction (TF-IDF)
    4. Model training (4 models compared)
    5. Evaluation (accuracy, F1, confusion matrix, ROC)
    6. Prediction on new tweets

  Requirements:
    pip install nltk scikit-learn pandas numpy matplotlib seaborn
=============================================================
"""

# ─────────────────────────────────────────────
#  1. IMPORTS
# ─────────────────────────────────────────────
import re
import string
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  2. DOWNLOAD NLTK RESOURCES
# ─────────────────────────────────────────────
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────
#  3. SAMPLE DATASET (no file needed to run)
# ─────────────────────────────────────────────
SAMPLE_DATA = {
    "text": [
        # Positive
        "I absolutely love this product! It changed my life 😍",
        "Great day today, feeling so happy and blessed! ☀️",
        "Just got promoted at work! Best day ever 🎉",
        "This movie was absolutely fantastic, highly recommend!",
        "The customer support was incredibly helpful and kind",
        "Had the best coffee this morning, feeling amazing!",
        "So proud of my team for crushing it today 💪",
        "Beautiful sunset tonight, nature is stunning 🌅",
        "Finally finished my project! So relieved and happy",
        "My dog just did the cutest thing ever 🐶❤️",
        "This book is a masterpiece, couldn't put it down",
        "Loving the new update, so many great features!",
        "Just had the most delicious meal of my life",
        "Incredible performance by the team last night! 🏆",
        "Feeling grateful for all the amazing people in my life",
        "New phone arrived today, the camera quality is insane!",
        "The concert was absolutely mind-blowing! Best night ever",
        "Woke up feeling energized and ready to conquer the day",
        "Just adopted a rescue puppy, I'm the happiest person alive",
        "Finally got my dream job offer! Hard work pays off",
        # Negative
        "This product is absolute garbage, total waste of money",
        "Worst customer service experience I've ever had. Never again!",
        "I'm so frustrated, nothing is going right today 😤",
        "This movie was a complete disappointment, boring and dull",
        "Stuck in traffic for 2 hours, absolutely terrible",
        "My laptop crashed and I lost all my work. Devastated 😢",
        "Can't believe how rude that person was to me",
        "This restaurant was disgusting, found a hair in my food",
        "Failed my exam after studying all night. I'm heartbroken",
        "The worst airline experience ever, lost my luggage",
        "So tired of all this negativity in the world 😞",
        "App keeps crashing, completely unusable. So angry right now",
        "Got caught in the rain without an umbrella. Miserable",
        "My order arrived broken. Terrible packaging and service",
        "Feeling so overwhelmed and stressed, can't handle this",
        "The internet is down again, this is so infuriating!",
        "Disappointed with the results, expected so much better",
        "Can't sleep, too anxious about tomorrow. This is awful",
        "Terrible weather ruining my plans for the whole week",
        "This update broke everything. The developers should be ashamed",
        # Neutral
        "Going to the grocery store after work today",
        "The weather forecast says rain tomorrow",
        "Just finished reading the news for today",
        "Attended a meeting this morning about Q3 goals",
        "The train arrives at 6:45 PM as scheduled",
        "Updating my phone software tonight",
        "Had pasta for lunch, now back to work",
        "The library closes at 8 PM on weekdays",
        "Watched the news before going to bed",
        "Waiting for my package to arrive this week",
    ],
    "sentiment": (
        ["positive"] * 20
        + ["negative"] * 20
        + ["neutral"] * 10
    ),
}


# ─────────────────────────────────────────────
#  4. DATA LOADING
# ─────────────────────────────────────────────
def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load dataset from a CSV file or fall back to the built-in sample.

    For Sentiment140 (https://www.kaggle.com/kazanova/sentiment140):
      - Columns: target, ids, date, flag, user, text
      - target: 0 = negative, 4 = positive

    For any custom CSV, expected columns: 'text', 'sentiment'
    ('positive', 'negative', or 'neutral')
    """
    if csv_path:
        df = pd.read_csv(csv_path, encoding="latin-1")

        # ── Sentiment140 format ──
        if "target" in df.columns and "ids" in df.columns:
            df = df[["target", "text"]]
            df["sentiment"] = df["target"].map({0: "negative", 4: "positive"})
            df = df[["text", "sentiment"]]
            # Use a balanced subset for speed
            pos = df[df.sentiment == "positive"].sample(5000, random_state=42)
            neg = df[df.sentiment == "negative"].sample(5000, random_state=42)
            df = pd.concat([pos, neg]).reset_index(drop=True)

        print(f"✅ Loaded dataset from CSV: {len(df):,} rows")
    else:
        df = pd.DataFrame(SAMPLE_DATA)
        print(f"✅ Using built-in sample dataset: {len(df)} rows")

    print(df["sentiment"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
#  5. TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """Full NLP cleaning pipeline for a single tweet."""
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove #hashtags (keep the word, drop the #)
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove RT prefix
    text = re.sub(r"\brt\b", "", text)
    # Remove emojis and non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace / digits
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize → remove stopwords → lemmatize
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔄 Preprocessing tweets...")
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.strip() != ""]  # drop empty rows
    print(f"   Cleaned {len(df):,} tweets.\n")
    return df


# ─────────────────────────────────────────────
#  6. FEATURE ENGINEERING + MODEL DEFINITIONS
# ─────────────────────────────────────────────
def build_models() -> dict:
    """Return a dict of sklearn Pipelines (TF-IDF + classifier)."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=50_000,
        sublinear_tf=True,    # log(1+TF) scaling
        min_df=2,
    )

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf),
            ("clf",  LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "Naive Bayes (MNB)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                                      sublinear_tf=False, min_df=2)),
            ("clf",  MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf),
            ("clf",  LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", tfidf),
            ("clf",  RandomForestClassifier(n_estimators=200, random_state=42,
                                             n_jobs=-1)),
        ]),
    }


# ─────────────────────────────────────────────
#  7. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    """Train all models and print full evaluation report."""
    X = df["clean_text"]
    y = df["sentiment"]
    classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    models = build_models()
    results = {}

    for name, pipeline in models.items():
        print(f"─── Training: {name} ───")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"  Accuracy : {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "report": report,
            "y_pred": y_pred,
            "y_test": y_test,
        }

    # ── Pick best model ──
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Best Model: {best_name} "
          f"(Accuracy: {results[best_name]['accuracy']:.4f})\n")

    return results, best_name, X_test, y_test, classes


# ─────────────────────────────────────────────
#  8. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_results(results: dict, best_name: str, classes: list,
                 df: pd.DataFrame):
    """Generate 4 plots: class dist, accuracy bar, confusion matrix, feature weights."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Twitter Sentiment Analysis — Results", fontsize=18,
                 fontweight="bold", y=0.98)

    # ── 1. Class distribution ──
    ax1 = fig.add_subplot(2, 2, 1)
    counts = df["sentiment"].value_counts()
    colors = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}
    bar_colors = [colors.get(c, "#9E9E9E") for c in counts.index]
    ax1.bar(counts.index, counts.values, color=bar_colors, edgecolor="white",
            linewidth=1.2)
    ax1.set_title("Class Distribution", fontsize=13, pad=10)
    ax1.set_ylabel("Count")
    for i, (val, col) in enumerate(zip(counts.values, counts.index)):
        ax1.text(i, val + 0.3, str(val), ha="center", fontsize=11)

    # ── 2. Model accuracy comparison ──
    ax2 = fig.add_subplot(2, 2, 2)
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    bar_clrs = ["#FFD700" if m == best_name else "#607D8B" for m in model_names]
    bars = ax2.barh(model_names, accuracies, color=bar_clrs, edgecolor="white")
    ax2.set_xlim(0, 1.05)
    ax2.set_title("Model Accuracy Comparison", fontsize=13, pad=10)
    ax2.set_xlabel("Accuracy")
    for bar, acc in zip(bars, accuracies):
        ax2.text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.4f}", va="center", fontsize=10)

    # ── 3. Confusion matrix (best model) ──
    ax3 = fig.add_subplot(2, 2, 3)
    best = results[best_name]
    cm = confusion_matrix(best["y_test"], best["y_pred"], labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax3,
                linewidths=0.5)
    ax3.set_title(f"Confusion Matrix — {best_name}", fontsize=13, pad=10)
    ax3.set_ylabel("True Label")
    ax3.set_xlabel("Predicted Label")

    # ── 4. Top features (best model if LR or SVM) ──
    ax4 = fig.add_subplot(2, 2, 4)
    pipeline = best["pipeline"]
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps["tfidf"]
    feature_names = np.array(tfidf.get_feature_names_out())

    try:
        if hasattr(clf, "coef_"):
            # Binary or multi-class
            if clf.coef_.shape[0] == 1:
                coef = clf.coef_[0]
                top_pos = np.argsort(coef)[-15:][::-1]
                top_neg = np.argsort(coef)[:15]
                top_idx = np.concatenate([top_pos, top_neg])
            else:
                # Multi-class: use positive class coef
                pos_idx = list(pipeline.classes_).index("positive") \
                    if hasattr(pipeline, "classes_") else 0
                coef = clf.coef_[pos_idx]
                top_pos = np.argsort(coef)[-10:][::-1]
                top_neg = np.argsort(coef)[:10]
                top_idx = np.concatenate([top_pos, top_neg])

            feat_coef = coef[top_idx]
            feat_labels = feature_names[top_idx]
            feat_colors = ["#4CAF50" if c > 0 else "#F44336" for c in feat_coef]
            ax4.barh(feat_labels, feat_coef, color=feat_colors)
            ax4.set_title(f"Top Features — {best_name}", fontsize=13, pad=10)
            ax4.set_xlabel("Coefficient Weight")
            ax4.axvline(0, color="black", linewidth=0.8)
        else:
            raise AttributeError
    except (AttributeError, IndexError):
        # Fallback: feature importance for RF
        if hasattr(clf, "feature_importances_"):
            top_idx = np.argsort(clf.feature_importances_)[-20:][::-1]
            ax4.barh(feature_names[top_idx],
                     clf.feature_importances_[top_idx], color="#607D8B")
            ax4.set_title(f"Feature Importances — {best_name}", fontsize=13)
        else:
            ax4.text(0.5, 0.5, "Feature plot N/A for this model",
                     ha="center", va="center", transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "/mnt/user-data/outputs/sentiment_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📊 Plot saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────
#  9. PREDICTION ON NEW TWEETS
# ─────────────────────────────────────────────
def predict_sentiment(pipeline, tweets: list[str]) -> pd.DataFrame:
    """
    Predict sentiment for a list of new tweets.

    Args:
        pipeline : trained sklearn Pipeline (best model)
        tweets   : list of raw tweet strings

    Returns:
        DataFrame with original text, cleaned text, prediction,
        and (if available) probability scores.
    """
    cleaned = [clean_tweet(t) for t in tweets]
    predictions = pipeline.predict(cleaned)

    rows = []
    clf = pipeline.named_steps["clf"]
    has_proba = hasattr(clf, "predict_proba")

    for raw, clean, pred in zip(tweets, cleaned, predictions):
        row = {"tweet": raw, "cleaned": clean, "sentiment": pred}
        if has_proba:
            proba = pipeline.predict_proba([clean])[0]
            for label, p in zip(pipeline.classes_, proba):
                row[f"prob_{label}"] = round(p, 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────
def main(csv_path: str = None):
    """
    Run the full pipeline.

    Usage:
        main()                          # uses built-in sample
        main("sentiment140.csv")        # uses your CSV file
    """
    print("=" * 60)
    print("  TWITTER SENTIMENT ANALYSIS — NLP + ML")
    print("=" * 60)

    # ── Load & Preprocess ──
    df = load_dataset(csv_path)
    df = preprocess(df)

    # ── Train & Evaluate ──
    results, best_name, X_test, y_test, classes = train_and_evaluate(df)

    # ── Plots ──
    plot_results(results, best_name, classes, df)

    # ── Predict on new tweets ──
    print("\n" + "=" * 60)
    print("  PREDICTING ON NEW TWEETS")
    print("=" * 60)

    new_tweets = [
        "Just bought the new iPhone and it's absolutely amazing! 🔥",
        "This traffic is unbearable, I'm going to be late again 😠",
        "Heading to the gym after work today",
        "The government should do more to fix the economy",
        "Can't believe how bad the service was at that restaurant!!",
        "Beautiful morning, feeling super motivated! #blessed",
        "My code finally works after debugging for 6 hours lol",
        "Watched a documentary last night, pretty informative",
    ]

    best_pipeline = results[best_name]["pipeline"]
    pred_df = predict_sentiment(best_pipeline, new_tweets)

    # Print nicely
    print(f"\n{'Tweet':<45} {'Sentiment':<12} {'Confidence'}")
    print("-" * 70)
    for _, row in pred_df.iterrows():
        tweet_short = row["tweet"][:44]
        sentiment = row["sentiment"].upper()
        conf = ""
        prob_col = f"prob_{row['sentiment']}"
        if prob_col in row:
            conf = f"{row[prob_col]*100:.1f}%"
        print(f"{tweet_short:<45} {sentiment:<12} {conf}")

    print("\n✅ Pipeline complete!")
    return results, best_pipeline, pred_df


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: use sample data (no file needed) ──
    results, best_model, predictions = main()

    # ── Option B: use Sentiment140 CSV from Kaggle ──
    # results, best_model, predictions = main("path/to/training.1600000.processed.noemoticon.csv")

    # ── Option C: use your own CSV (must have 'text' + 'sentiment' columns) ──
    # results, best_model, predictions = main("my_tweets.csv")

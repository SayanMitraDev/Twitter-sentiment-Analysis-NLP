from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Positive 😊" if pred == 1 else "Negative 😡"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form["tweet"]
        result = predict_sentiment(tweet)

    return render_template("index.html", result=result, tweet=tweet)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment_scores = sia.polarity_scores(text)
    return render_template('result.html', text=text, sentiment=sentiment_scores)

if __name__ == '__main__':
    app.run(debug=True)

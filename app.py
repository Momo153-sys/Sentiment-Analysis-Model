from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import os

# Ensure VADER lexicon is available (safe for production)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize app
app = Flask(__name__)
CORS(app)

# Initialize analyzer once (not per request)
sia = SentimentIntensityAnalyzer()


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json(silent=True)

    if not data or 'text' not in data:
        return jsonify({
            'error': 'Text field is required'
        }), 400

    text = data['text']
    scores = sia.polarity_scores(text)

    # Sentiment classification
    compound = scores.get('compound', 0)

    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'scores': scores
    })


@app.route('/')
def health_check():
    return jsonify({'status': 'API is running'}), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port)

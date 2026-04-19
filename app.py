from flask import Flask, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS
import nltk

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    
    text = data['text']
    scores = sia.polarity_scores(text)
    
    # Determine sentiment based on compound score
    if scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'scores': scores
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)

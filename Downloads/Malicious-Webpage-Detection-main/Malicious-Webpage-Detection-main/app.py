from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from url_features import URLFeatureExtractor

app = Flask(__name__)

# Load trained model and components
model = joblib.load('url_classifier_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_extractor = joblib.load('feature_extractor.pkl')

# Risk display info for binary classes
RISK_LEVELS = {
    'benign': {
        'color': 'success',
        'icon': 'fa-check-circle',
        'level': 'Safe'
    },
    'malicious': {
        'color': 'danger',
        'icon': 'fa-skull-crossbones',
        'level': 'Malicious'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data['url']

        # Extract features
        features = feature_extractor.extract_features(url)

        # Predict
        prediction_idx = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        # Decode label
        pred_label = label_encoder.inverse_transform([prediction_idx])[0]

        # Get risk info
        risk_info = RISK_LEVELS.get(pred_label, RISK_LEVELS['malicious'])

        # Map class probabilities to labels (only 'benign' and 'malicious')
        class_probs = dict(zip(label_encoder.classes_, probabilities))

        return jsonify({
            'success': True,
            'prediction': pred_label,
            'confidence': float(np.max(probabilities)),
            'risk_level': risk_info['level'],
            'risk_color': risk_info['color'],  # success or danger
            'icon': risk_info['icon'],         # font-awesome icon class
            'probabilities': class_probs,
            'url': url
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
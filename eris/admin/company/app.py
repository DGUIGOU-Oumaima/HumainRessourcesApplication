from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Charger le modèle pré-entraîné (remplacez 'ensemble_model.pkl' par le chemin de votre modèle)
model = joblib.load('D:/xampp/htdocs/eris/admin/company/ensemble_model.pkl')

@app.route('/')
def index():
    return render_template('home.php')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file:
        df = pd.read_csv(file)
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df).tolist()
        formatted_probabilities = [[f"{prob:.2f}" for prob in prob_list] for prob_list in probabilities]
        results = [{'prediction': int(pred), 'probabilities': prob} for pred, prob in zip(predictions, formatted_probabilities)]
        return jsonify(results)
    return jsonify({'error': 'File processing error'})

if __name__ == '__main__':
    app.run(debug=True)

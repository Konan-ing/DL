from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app) # Autorise votre portfolio à contacter l'API

# Charger le modèle au démarrage
model = tf.keras.models.load_model('mon_modele_ia.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        # Préparation de la donnée pour le modèle
        input_data = np.array([[float(data)]], dtype=float)
        
        # Prédiction
        prediction = model.predict(input_data)
        resultat = float(prediction[0][0])
        
        return jsonify({"prediction": round(resultat, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
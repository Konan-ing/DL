from flask import Flask, request, jsonify
from flask_cors import CORS # <--- INDISPENSABLE
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# 1. Gestion du CORS : Autorise votre navigateur à appeler l'API
# En local, on autorise tout pour ne pas être bloqué
CORS(app) 

# 2. Chargement du modèle
model = tf.keras.models.load_model('mon_modele_ia.h5')

# 3. Définition de la route
@app.route('/predict', methods=['POST'])
@app.route("/")
def home():
    return "API is running"

def predict():
    data = request.get_json()

    # Récupération de la valeur envoyée
    input_value = data.get("input")

    # Si l'utilisateur envoie un scalaire
    if isinstance(input_value, (int, float)):
        input_array = np.array([[float(input_value)]], dtype=float)

    # Si l'utilisateur envoie un tableau 1D : [1,2,3]
    elif isinstance(input_value, list) and all(isinstance(x, (int, float)) for x in input_value):
        input_array = np.array(input_value, dtype=float).reshape(-1, 1)

    # Si l'utilisateur envoie un tableau 2D : [[1],[2],[3]]
    elif isinstance(input_value, list) and all(isinstance(x, list) for x in input_value):
        input_array = np.array(input_value, dtype=float)

    else:
        return jsonify({"error": "Format d'entrée invalide"}), 400

    # Prédiction
    prediction = model.predict(input_array)

    return jsonify({"prediction": prediction.tolist()})


if __name__ == '__main__':
    # Lancement du serveur sur le port 5000
    app.run(port=5000, debug=True)
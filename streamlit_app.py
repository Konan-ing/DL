import streamlit as st
import numpy as np
import tensorflow as tf

st.title("Mini modèle TensorFlow Lite – Démo ultra rapide")

# Charger le modèle TFLite une seule fois
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Fonction de prédiction TFLite
def predict_tflite(value):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = np.array([[value]], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output.tolist()

# Interface utilisateur
value = st.number_input("Entrer une valeur (0 à 1)", min_value=0.0, max_value=1.0, step=0.01)

if st.button("Prédire"):
    result = predict_tflite(value)
    st.success(f"Résultat : {result}")

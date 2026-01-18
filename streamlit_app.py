import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Mini modèle TensorFlow – Démo rapide")

# Chargement du modèle avec cache (super important)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mon_modele_ia.h5")

model = load_model()

# Interface utilisateur
value = st.number_input("Entrer une valeur (0 à 1)", min_value=0.0, max_value=1.0, step=0.01)

if st.button("Prédire"):
    arr = np.array([[value]], dtype=float)
    pred = model.predict(arr)
    st.success(f"Résultat : {pred.tolist()}")

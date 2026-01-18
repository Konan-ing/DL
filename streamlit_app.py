import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Interface de test du modèle")

model = tf.keras.models.load_model("mon_modele_ia.h5")

value = st.number_input("Entrer une valeur")

if st.button("Prédire"):
    arr = np.array([[value]], dtype=float)
    pred = model.predict(arr)
    st.write("Résultat :", pred.tolist())

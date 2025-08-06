import base64
import streamlit as st
import numpy as np
import joblib

# === Estilos con fondo y overlay ===
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                              url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .title-container {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 15px;
        }}
        .title-text {{
            color: white;
            font-size: 34px;
            font-weight: bold;
        }}
        .subtitle-text {{
            color: white;
            font-size: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# === Cargar modelos ===
@st.cache_resource
def cargar_modelos():
    ruta = "."
    scaler = joblib.load(f"{ruta}/scaler.pkl")
    kmeans = joblib.load(f"{ruta}/kmeans.pkl")
    modelos = {i: joblib.load(f"{ruta}/modelo_cluster_{i}_General.pkl") for i in range(4)}
    return scaler, kmeans, modelos

nombres_cluster = {
    0: "Gran T

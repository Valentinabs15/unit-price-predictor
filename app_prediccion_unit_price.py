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

# Diccionario de nombres de clúster corregido
nombres_cluster = {
    0: "Gran Tronadura",
    1: "Tronadura Fuerte",
    2: "Tronadura Intermedia",
    3: "Tronadura Estándar"
}

zonas = ['Centro', 'Norte Chico', 'Norte Grande', 'Sur']
campos = [
    "CLP/USD",
    "IPC CL",
    "FERT AMM",
    "FERTECON",
    "CPI USA",
    "ENAP Diesel"
]

# === Función de predicción ===
def predecir_precio(volume, zona_str, valores_macro, scaler, kmeans, modelos):
    zona_id = zonas.index(zona_str)

    # Input para clustering
    cluster_input = np.array([[volume, zona_id] + valores_macro])
    cluster_scaled = scaler.transform(cluster_input)
    cluster = kmeans.predict(cluster_scaled)[0]
    nombre_cluster = nombres_cluster.get(cluster, f"Cluster {cluster}")

    modelo = modelos.get(cluster)
    if modelo is None:
        return None

    # Input para predicción final
    X_pred = np.array([volume] + valores_macro).reshape(1, -1)
    precio_estimado = modelo.predict(X_pred)[0]

    return cluster, nombre_cluster, precio_estimado

# === App ===
set_background("fondo.jpg")

# Título y subtítulo
st.markdown(
    '<div class="title-container">'
    '<div class="title-text">🔍 Predicción de Precio Unitario</div>'
    '<div class="subtitle-text">Simula escenarios por zona y volumen con variables macroeconómicas</div>'
    '</div>',
    unsafe_allow_html=True
)

# Cargar modelos
scaler, kmeans, modelos = cargar_modelos()

# Entradas
volume = st.number_input("📦 Volumen (toneladas)", min_value=0.0, value=200.0)
zona = st.selectbox("📍 Zona", zonas)

st.markdown("### 📊 Variables Macroeconómicas")

# Etiquetas con estilo mejorado
valores_macro = []
for campo in campos:
    st.markdown(
        f"<span style='color:white;font-weight:bold;background-color:rgba(0,0,0,0.5);padding:4px;border-radius:4px'>{campo}</span>",
        unsafe_allow_html=True
    )
    val = st.number_input("", value=0.0, key=campo)
    valores_macro.append(val)

# Botón de predicción
if st.button("📈 Predecir"):
    resultado = predecir_precio(volume, zona, valores_macro, scaler, kmeans, modelos)
    if resultado:
        cluster_id, cluster_nombre, prediccion = resultado
        st.success(f"📌 Cluster: {cluster_id} - {cluster_nombre}")
        st.write(f"📊 Precio proyectado: **${prediccion:.2f} AUD/Ton**")
    else:
        st.error("❌ No se encontró modelo para el clúster.")

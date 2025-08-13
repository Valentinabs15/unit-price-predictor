import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --- 1. Cargar modelos y definir mapeos ---
@st.cache_resource
def cargar_modelos_y_definir():
    """Carga los modelos, escaladores y define los mapeos."""
    try:
        # Cargar los archivos necesarios
        scaler_cluster = joblib.load("scaler_cluster.pkl")
        kmeans = joblib.load("kmeans.pkl")
        modelos = {i: joblib.load(f"modelo_cluster_{i}_General.pkl") for i in range(4)}
        
        # Mapeo de nombres de clúster a sus IDs
        nombres_cluster = {
            0: "Gran Tronadura",
            1: "Tronadura Fuerte",
            2: "Tronadura Intermedia",
            3: "Tronadura Estándar"
        }
        
        # Mapeo de zona para el clustering
        zonas_map = {
            'Gran Tronadura': 0,
            'Tronadura Fuerte': 1,
            'Tronadura Intermedia': 2,
            'Tronadura Estándar': 3
        }
        
        campos_modelo = [
            "Volume", "Zone", "CLP_USD_FX_MONTHLY", "IPC_BASE2018_CP_CL_MONTHLY",
            "FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY", "FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP",
            "CPI_USA_MONTHLY", "FUENTE_ENAP_CHL_DL_MONTHLY"
        ]
        
        return scaler_cluster, kmeans, modelos, nombres_cluster, zonas_map, campos_modelo

    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar archivos del modelo: {e.filename}. Asegúrate de que todos los archivos .pkl están en la misma carpeta.")
        return None, None, None, None, None, None

# --- 2. Función de Predicción ---
def predecir_varios_periodos(volume, zona_str, lista_valores_por_periodo, scaler_cluster, kmeans, modelos, nombres_cluster, zonas_map, campos_modelo):
    """Realiza la predicción para múltiples períodos."""
    
    # Obtener el ID numérico de la zona/clúster
    zona_id = zonas_map.get(zona_str)
    if zona_id is None:
        st.error(f"Zona '{zona_str}' no es válida.")
        return None

    # Asignar el clúster usando el modelo KMeans
    cluster_input = pd.DataFrame([[volume, zona_id]], columns=['Volume', 'Zone'])
    cluster_scaled = scaler_cluster.transform(cluster_input)
    cluster = kmeans.predict(cluster_scaled)[0]
    nombre_cluster = nombres_cluster.get(cluster, f"Clúster {cluster}")
    
    modelo = modelos.get(cluster)
    if modelo is None:
        return None

    predicciones = []
    for valores in lista_valores_por_periodo:
        # Corregido para construir el DataFrame de entrada completo para el modelo
        data_pred = pd.DataFrame([[volume, zona_id] + valores], columns=campos_modelo)
        
        # El pipeline del modelo se encarga del escalado automáticamente
        precio_estimado = modelo.predict(data_pred)[0]
        predicciones.append(precio_estimado)

    return cluster, nombre_cluster, predicciones

# --- 3. Interfaz Streamlit ---
st.title("Predicción de Precio Unitario por Cliente")
st.markdown("Basado en modelos segmentados por clúster y variables macroeconómicas")

scaler_cluster, kmeans, modelos, nombres_cluster, zonas_map, campos_modelo = cargar_modelos_y_definir()

if not all([scaler_cluster, kmeans, modelos, nombres_cluster, zonas_map, campos_modelo]):
    st.stop()

# Campos de entrada
st.markdown("---")
st.markdown("### 1. Datos del Cliente")
col1, col2 = st.columns(2)
with col1:
    volume = st.number_input("Volumen (ton)", min_value=0.0, value=200.0)
with col2:
    zona = st.selectbox("Zona del cliente", options=list(zonas_map.keys()))

st.markdown("---")
st.markdown("### 2. Variables Económicas por Período")
periodos = st.number_input("¿Cuántos períodos quieres predecir?", min_value=1, max_value=12, value=1)

lista_valores = []
campos_economicos = campos_modelo[2:]
for i in range(periodos):
    st.subheader(f"Período {i+1}")
    valores = []
    cols = st.columns(3)
    for idx, campo in enumerate(campos_economicos):
        with cols[idx % 3]:
            val = st.number_input(f"{campo}", value=0.0, key=f"p{i+1}_{campo}")
            valores.append(val)
    lista_valores.append(valores)
    st.markdown("---")

# Botón de predicción
if st.button("Predecir", use_container_width=True):
    with st.spinner('Calculando predicción...'):
        resultado = predecir_varios_periodos(volume, zona, lista_valores, scaler_cluster, kmeans, modelos, nombres_cluster, zonas_map, campos_modelo)
    
    if resultado:
        cluster_id, cluster_nombre, predicciones = resultado
        st.success(f"📌 **Clúster asignado:** {cluster_nombre} (ID: {cluster_id})")
        st.markdown("### Predicciones por Período")
        for i, pred in enumerate(predicciones):
            st.write(f"🗓️ **Período {i+1}:** **${pred:,.2f}**")
    else:
        st.error("No se pudo realizar la predicción. Revisa los datos de entrada y que los modelos estén cargados correctamente.")

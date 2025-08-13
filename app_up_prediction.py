import sys
import traceback
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64

# =============================
# Configuración de la página
# =============================
st.set_page_config(page_title="Predicción de Precio Unitario", page_icon="💥", layout="wide")
st.title("💥 Predicción de Precio Unitario por Cliente")
st.caption("Alineado al notebook: zonas geográficas correctas y fondo opcional.")

# Fondo por defecto (gradiente suave) si no se aporta imagen/URL
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, rgba(245,246,250,1) 0%, rgba(255,255,255,1) 60%);
}
</style>
""", unsafe_allow_html=True)

# =============================
# Parámetros y archivos esperados
# =============================
DEFAULT_MODELS_DIR = Path(__file__).parent / "modelos"

EXPECTED = {
    "scaler_cluster": "scaler_cluster.pkl",
    "kmeans": "kmeans.pkl",
    "modelo_cluster_0": "modelo_cluster_0_General.pkl",
    "modelo_cluster_1": "modelo_cluster_1_General.pkl",
    "modelo_cluster_2": "modelo_cluster_2_General.pkl",
    "modelo_cluster_3": "modelo_cluster_3_General.pkl",
}

# Columnas del modelo (ORDEN IMPORTA)
CAMPOS_MODELO = [
    "Volume", "Zone", "CLP_USD_FX_MONTHLY", "IPC_BASE2018_CP_CL_MONTHLY",
    "FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY", "FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP",
    "CPI_USA_MONTHLY", "FUENTE_ENAP_CHL_DL_MONTHLY"
]

# Etiquetas legibles para el clúster (solo display)
NOMBRES_CLUSTER = {
    0: "Gran Tronadura",
    1: "Tronadura Fuerte",
    2: "Tronadura Intermedia",
    3: "Tronadura Estándar"
}

# Zonas según el notebook (NO editar: deben mapearse igual que en el entrenamiento)
ZONAS_MAP = {'Centro': 0, 'Norte Chico': 1, 'Norte Grande': 2, 'Sur': 3}

# =============================
# Apariencia: fondo personalizado (opcional)
# =============================
def set_background(image_bytes: bytes = None, image_url: str = None):
    css = ""
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        css = f"""
        <style>
        .stApp {
            background-image: url('data:image/png;base64,{b64}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """
    elif image_url:
        css = f"""
        <style>
        .stApp {
            background-image: url('{image_url}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """
    if css:
        st.markdown(css, unsafe_allow_html=True)


# ===== Fondo desde GitHub/local =====
def normalize_github_url(url: str) -> str:
    """Convierte un link 'github.com/.../blob/...' a 'raw.githubusercontent.com/...'."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url

def try_load_local_background():
    # Busca 'fondo' con extensiones comunes en ./assets o en la carpeta del app
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    bases = [Path(__file__).parent / "assets", Path(__file__).parent]
    for base in bases:
        for ext in exts:
            f = base / f"fondo{ext}"
            if f.exists():
                try:
                    return f.read_bytes()
                except Exception:
                    pass
    return None

# =============================
# Utilidades
# =============================
def check_required_files(models_dir: Path):
    missing = []
    paths = {}
    for key, fname in EXPECTED.items():
        p = models_dir / fname
        paths[key] = p
        if not p.exists():
            missing.append(str(p))
    return missing, paths

@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir: Path):
    models_dir = Path(models_dir)
    missing, paths = check_required_files(models_dir)
    if missing:
        raise FileNotFoundError("Faltan archivos de modelo:\n- " + "\n- ".join(missing))

    scaler_cluster = joblib.load(paths["scaler_cluster"])
    kmeans = joblib.load(paths["kmeans"])
    modelos = {
        0: joblib.load(paths["modelo_cluster_0"]),
        1: joblib.load(paths["modelo_cluster_1"]),
        2: joblib.load(paths["modelo_cluster_2"]),
        3: joblib.load(paths["modelo_cluster_3"]),
    }
    return scaler_cluster, kmeans, modelos

def predecir_varios_periodos(volume, zona_str, lista_valores_por_periodo,
                             scaler_cluster, kmeans, modelos, zonas_map):
    # Validación de zona según notebook
    if zona_str not in zonas_map:
        raise ValueError(f"Zona '{zona_str}' no es válida. Debe ser una de: {list(zonas_map.keys())}")
    zona_id = zonas_map[zona_str]

    # Clusterización con Volume y Zone (mismo orden y tipos que en el entrenamiento)
    X_cluster = pd.DataFrame([[volume, zona_id]], columns=["Volume", "Zone"])
    X_cluster_scaled = scaler_cluster.transform(X_cluster)
    cluster = int(kmeans.predict(X_cluster_scaled)[0])

    # Predicciones por período
    model = modelos.get(cluster)
    if model is None:
        raise RuntimeError(f"No hay modelo disponible para el clúster {cluster}.")

    preds = []
    for valores in lista_valores_por_periodo:
        fila = [volume, zona_id] + list(valores)
        X = pd.DataFrame([fila], columns=CAMPOS_MODELO)
        y_hat = float(model.predict(X)[0])
        preds.append(y_hat)

    return cluster, preds

# =============================
# Sidebar: configuración, verificación y fondo
# =============================
st.sidebar.header("⚙️ Configuración")
models_dir_str = st.sidebar.text_input("Carpeta de modelos (.pkl)", value=str(DEFAULT_MODELS_DIR))
models_dir = Path(models_dir_str)

with st.sidebar.expander("Verificación de archivos", expanded=False):
    miss, _ = check_required_files(models_dir)
    if miss:
        st.warning("Archivos faltantes:")
        for m in miss:
            st.code(m, language="text")
    else:
        st.success("Todos los archivos requeridos están presentes.")

try:
    scaler_cluster, kmeans, modelos = load_artifacts(models_dir)
    st.sidebar.success("Modelos cargados correctamente.")
except Exception as e:
    st.sidebar.error("No se pudieron cargar los modelos.")
    st.sidebar.code(f"{e}\n\n{traceback.format_exc()}", language="text")
    st.stop()

with st.sidebar.expander("Fondo de la app (GitHub / local)", expanded=True):
    st.caption("La app intentará usar automáticamente un archivo local llamado **fondo.png/jpg**. O pega el link de GitHub.")
    bg_local = try_load_local_background()
    if bg_local:
        set_background(bg_local, None)
        st.success("Fondo local \"fondo\" aplicado.")
    bg_url = st.text_input("URL de imagen (GitHub o RAW)", value="", placeholder="https://github.com/usuario/repo/blob/main/fondo.png")
    if bg_url:
        set_background(None, normalize_github_url(bg_url))
        st.info("Fondo desde URL aplicado.")

# =============================
# UI de entrada
# =============================
st.markdown("---")
st.subheader("1) Datos del cliente")
st.caption("Zonas según el notebook → {ZONAS_MAP} (se mapean al entero de la columna 'Zone').")

col1, col2 = st.columns(2)
with col1:
    volume = st.number_input("Volumen (ton)", min_value=0.0, value=200.0, step=1.0)
with col2:
    zona = st.selectbox("Zona geográfica", options=list(ZONAS_MAP.keys()))

st.markdown("---")
st.subheader("2) Variables económicas por período")
periodos = st.number_input("¿Cuántos períodos quieres predecir?", min_value=1, max_value=12, value=1, step=1)

lista_valores = []
campos_economicos = CAMPOS_MODELO[2:]
for i in range(int(periodos)):
    st.markdown(f"**Período {i+1}**")
    cols = st.columns(3)
    valores = []
    for j, campo in enumerate(campos_economicos):
        with cols[j % 3]:
            val = st.number_input(f"{campo} (P{i+1})", value=0.0, key=f"p{i+1}_{campo}")
            valores.append(val)
    lista_valores.append(valores)
    st.divider()

# =============================
# Botón de predicción
# =============================
if st.button("Predecir", use_container_width=True):
    try:
        with st.spinner("Calculando predicciones..."):
            cluster, preds = predecir_varios_periodos(
                volume, zona, lista_valores, scaler_cluster, kmeans, modelos, ZONAS_MAP
            )

        nombre_cluster = NOMBRES_CLUSTER.get(cluster, f"Clúster {cluster}")
        st.success(f"📌 Clúster asignado: **{nombre_cluster}** (ID: {cluster})")

        df_res = pd.DataFrame({
            "Período": [f"P{i+1}" for i in range(len(preds))],
            "Precio Predicho": [round(x, 4) for x in preds]
        })
        st.subheader("Resultados")
        st.dataframe(df_res, use_container_width=True)
        st.metric(label="Precio (Período 1)", value=f"{preds[0]:,.4f}")

    except Exception as e:
        st.error("Ocurrió un error durante la predicción.")
        st.code(f"{e}\n\n{traceback.format_exc()}", language="text")

# =============================
# Diagnóstico
# =============================
with st.expander("🔍 Diagnóstico"):
    import platform, sklearn
    st.write("Python:", sys.version.split()[0])
    st.write("Sistema:", platform.platform())
    st.write("scikit-learn:", sklearn.__version__)
    st.write("joblib:", joblib.__version__)
    st.write("Directorio de modelos:", str(models_dir.resolve()))
    st.write("Archivos esperados:", EXPECTED)
    st.write("Campos del modelo:", CAMPOS_MODELO)
    st.write("Zonas (según notebook):", ZONAS_MAP)

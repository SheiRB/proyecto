
import streamlit as st
import pandas as pd
import joblib
import base64

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="Netflix Type Predictor",
    page_icon="🎬",
    layout="wide"
)

# =========================
# ESTILOS CSS - NETFLIX
# =========================
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #141414;
    color: white;
    font-family: 'Helvetica Neue', sans-serif;
}

.main {
    background-color: #141414;
}

h1, h2, h3 {
    color: #E50914;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #b20710;
    transform: scale(1.03);
}

.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    background-color: #333333;
    color: white;
    border-radius: 8px;
    border: 1px solid #555;
}

.card {
    background-color: #1f1f1f;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(229,9,20,0.3);
}

.small-text {
    font-size: 15px;
    color: #bbbbbb;
}

.result-box {
    background-color: #E50914;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# CARGAR MODELOS
# =========================
logistic_model = joblib.load("modelos/logistic_regression_model.pkl")
random_forest_model = joblib.load("modelos/random_forrest_model.pkl")

# =========================
# HEADER
# =========================
st.markdown("""
# 🎬 NETFLIX TYPE PREDICTOR
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="card">
        <h3>👩‍💻 Información</h3>
        <p><b>Nombre:</b> Sheila</p>
        <p><b>Código:</b> 40558361</p>
        <p class="small-text">
        🔗 Colab:
        <a href="https://colab.research.google.com/drive/1lZCBEIANlynDQTNQnJ-Y3Mdz0P1i9bSm#scrollTo=qc570ki3CBhG" target="_blank">
        Abrir Notebook
        </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# FORMULARIO
# =========================
st.markdown("## 📋 Ingrese los datos")

with st.container():
    
    title = st.text_input("🎞️ Title")
    director = st.text_input("🎬 Director")
    country = st.text_input("🌍 Country")
    release_year = st.number_input(
        "📅 Release Year",
        min_value=1900,
        max_value=2030,
        value=2020
    )
    
    duration = st.number_input(
        "⏳ Duration",
        min_value=1,
        max_value=500,
        value=90
    )

# =========================
# SELECCIÓN DE MODELO
# =========================
model_option = st.selectbox(
    "🤖 Seleccione el modelo",
    ("Logistic Regression", "Random Forest")
)

# =========================
# PREDICCIÓN
# =========================
if st.button("🔮 Obtener Predicción"):

    # DataFrame con variables necesarias
    # Solo usamos duration porque el modelo fue entrenado con duración
    
    input_data = pd.DataFrame({
        "duration": [duration]
    })

    # Seleccionar modelo
    if model_option == "Logistic Regression":
        prediction = logistic_model.predict(input_data)[0]
    else:
        prediction = random_forest_model.predict(input_data)[0]

    # Resultado
    st.markdown(f"""
    <div class="result-box">
        Predicción: {prediction}
    </div>
    """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
---
<center>
<p style='color:gray'>
Netflix Style Streamlit App 🎥
</p>
</center>
""", unsafe_allow_html=True)

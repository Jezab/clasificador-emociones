import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Configuración de página
st.set_page_config(
    page_title="Clasificador de emociones con IA",
    layout="centered"
)

# 2. Estilos visuales modernos
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .main-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(90deg, #4A00E0, #8E2DE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #6c757d;
        font-size: 16px;
        margin-bottom: 25px;
    }
    .result-card {
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        border-radius: 16px;
        padding: 22px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        margin-top: 25px;
        margin-bottom: 20px;
    }
    .result-emotion {
        font-size: 26px;
        font-weight: bold;
        color: #00F2FE;
        margin-bottom: 8px;
    }
    .result-confidence {
        font-size: 18px;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Encabezado
st.markdown("""
<div class="main-header">
    <div class="main-title"> Clasificador de emociones con IA</div>
    <div class="subtitle">Sube una imagen y descubre la emoción que refleja el rostro</div>
</div>
""", unsafe_allow_html=True)

# 4. Cargar modelo por pesos nativos
@st.cache_resource
def load_emotion_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.load_weights("pesos_emociones.weights.h5")
    return model

try:
    modelo = load_emotion_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 4 Emociones
EMOCIONES = ["Enojado 😡", "Feliz 😄", "Neutral 😐", "Sorprendido 😲"]

# 5. Entrada para subir fotos
uploaded_file = st.file_uploader(
    "📁 Sube una imagen con rostro (JPG, JPEG o PNG)", 
    type=["jpg", "jpeg", "png"]
)

# 6. Procesamiento y Clasificación directa con PIL
if uploaded_file is not None:
    # Cargar imagen original
    img_original = Image.open(uploaded_file).convert("RGB")

    # Preprocesar a escala de grises y tamaño 48x48
    img_gray = ImageOps.grayscale(img_original)
    img_resized = img_gray.resize((48, 48))

    # Normalizar para la CNN
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.reshape(img_array, (1, 48, 48, 1))

    # Predicción con la red neuronal
    prediccion = modelo.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(prediccion))
    emocion_top = EMOCIONES[idx]
    certeza_top = float(prediccion[idx]) * 100

    # Mostrar imagen centrada
    st.image(img_original, caption="Imagen Analizada", use_container_width=True)

    # Tarjeta de resultado principal
    st.markdown(f"""
    <div class="result-card">
        <div class="result-emotion">Emoción Detectada: {emocion_top}</div>
        <div class="result-confidence">Certeza: <strong>{certeza_top:.2f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Desglose de porcentajes interactivo
    st.subheader("Desglose de Probabilidades")
    for i, emo in enumerate(EMOCIONES):
        prob = float(prediccion[i])
        st.write(f"**{emo}**: {prob * 100:.1f}%")
        st.progress(prob)

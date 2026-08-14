import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Configuración de página
st.set_page_config(
    page_title="Clasificador de emociones con IA",
    page_icon="",
    layout="centered"
)

# 2. Estilos personalizados modernos
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
    <div class="main-title">Detector de emociones con IA</div>
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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 5. Entrada exclusiva para subir fotos
uploaded_file = st.file_uploader(
    "Sube una imagen con rostro (JPG, JPEG o PNG)", 
    type=["jpg", "jpeg", "png"]
)

# 6. Procesamiento y Clasificación
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    img_original = img_np.copy()
    img_marked = img_np.copy()

    # Detección de rostros con Haar Cascades
    caras = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(caras) == 0:
        st.warning("No se detectó ningún rostro en la imagen. Intenta con una foto más clara o mejor iluminada.")
    else:
        emocion_top = ""
        certeza_top = 0.0
        all_preds = []

        for (x, y, w, h) in caras:
            # Recortar área del rostro y preprocesar a 48x48
            rostro = gray[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (48, 48))
            rostro = rostro.astype("float32") / 255.0
            rostro = np.expand_dims(rostro, axis=0)
            rostro = np.expand_dims(rostro, axis=-1)

            # Inferencia con la CNN
            prediccion = modelo.predict(rostro, verbose=0)[0]
            all_preds = prediccion
            idx = int(np.argmax(prediccion))
            emocion_top = EMOCIONES[idx]
            certeza_top = float(prediccion[idx]) * 100

            # Dibujar recuadro cian y texto de la emoción sobre el rostro
            cv2.rectangle(img_marked, (x, y), (x + w, y + h), (0, 242, 254), 3)
            cv2.putText(img_marked, emocion_top.split()[0], (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 242, 254), 2)

        # Mostrar imágenes lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="Imagen Original", use_container_width=True)
        with col2:
            st.image(img_marked, caption="Rostro Detectado", use_container_width=True)

        # Tarjeta de resultado principal
        st.markdown(f"""
        <div class="result-card">
            <div class="result-emotion">Emoción Detectada: {emocion_top}</div>
            <div class="result-confidence">Certeza: <strong>{certeza_top:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Desglose de porcentajes
        st.subheader("Desglose de Probabilidades")
        for i, emo in enumerate(EMOCIONES):
            prob = float(all_preds[i])
            st.write(f"**{emo}**: {prob * 100:.1f}%")
            st.progress(prob)

import os
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

# 1. Cargar modelo .h5 de forma segura
model = tf.keras.models.load_model("modelo_emociones.h5", compile=False)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EMOTIONS = ["Enojado 😡", "Feliz 😄", "Neutral 😐", "Sorprendido 😲"]


def predict_emotion(image):
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        roi = cv2.resize(gray, (48, 48))
    else:
        (x, y, w, h) = faces[0]
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (48, 48))

    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)

    preds = model.predict(roi, verbose=0)[0]
    return {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}


# 2. Interfaz visual
custom_theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="pink")

with gr.Blocks(theme=custom_theme, title="Clasificador de Emociones") as app:
    gr.Markdown(
        """
        # Clasificador de Emociones
        Detecta 4 emociones clave (Enojado, Feliz, Neutral y Sorpresa).
        """
    )
    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Cámara / Imagen",
            )
            btn = gr.Button("Analizar Emoción ", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=4, label="Resultado")

    btn.click(fn=predict_emotion, inputs=webcam_input, outputs=label_output)

# 3. Lanzador con variables de entorno para Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=False,
    )

import os
import cv2
import gradio as gr
import keras  # <-- Usamos Keras 3 directamente
import numpy as np

# Cargar el modelo con Keras 3
model = keras.models.load_model("modelo_emociones.keras")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 4 Emociones
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


# Interfaz de Gradio
custom_theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="pink")

with gr.Blocks(theme=custom_theme, title="Clasificador de Emociones AI") as app:
    gr.Markdown(
        """
        # Clasificador de Emociones en Tiempo Real
        ### Detecta 4 emociones clave (Enojado, Feliz, Neutral y Sorpresa) usando una CNN.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Cámara / Imagen",
            )
            btn = gr.Button("Analizar Emoción ", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=4, label="Emoción Detectada")

    btn.click(fn=predict_emotion, inputs=webcam_input, outputs=label_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)

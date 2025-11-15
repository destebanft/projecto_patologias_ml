from flask import Flask, render_template, request, redirect, jsonify
import datetime
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
# from tensorflow.keras.utils import img_to_array
# from tensorflow.keras.preprocessing import image
import io
# import torch
# import torchvision.transforms as T
import keras

# tf.config.set_visible_devices([], 'GPU')

model_rd = keras.models.load_model(
    'model/resnet_model_fixed.keras',
    compile=False,
    safe_mode=False)

# model_yolo = torch.jit.load('model/best.torchscript')


app = Flask(__name__)
CORS(app)


def prepare_image_for_prediction(image_path, target_size=(224, 224)):
    """
    Prepara una imagen individual para ser usada por el modelo.
    """
    img = Image.open(image_path)

    # Convertir a RGB si no lo está (importante si el modelo espera 3 canales)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Redimensionar la imagen al tamaño esperado por el modelo
    img = img.resize(target_size)

    # Convertir la imagen a un array de NumPy
    img_array = np.array(img)

    # Expandir las dimensiones para que coincida con el formato de entrada del modelo (batch_size, height, width, channels)
    # El modelo espera un "batch" de imágenes, incluso si es solo una.
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/")
def hello_world():
    return "Hello, World from Flask!"


@app.route('/predict_rd', methods=['GET', 'POST'])
def predict_rd():
    categories = {
        0: "*Recomendación: Limpiar, proteger el acero con un inhibidor de corrosión y recubrir con mortero de reparación.",
        1: "*Recomendación:  Sellar con material elástico o inyectar resina epoxi, según el tipo y movimiento de la grieta",
        2: "*Recomendación:  Limpiar la zona afectada y rellenar con mortero de reparación de alta adherencia y sin contracción"
    }
    if request.method == 'GET':
        return render_template('prediction_rd.html')
    if request.method == 'POST':
        print('post.......')
        try:
            # Verificar si la imagen fue enviada en el request
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded"}), 400

            # Leer la imagen desde el request
            image_file = request.files["image"]

            # Abrir la imagen con PIL desde el objeto de archivo
            img = Image.open(image_file.stream)

            # Convertir a RGB si no lo está y redimensionar
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))

            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Asumiendo que 'model_rd' ya está cargado y listo para predicción
            prediction = model_rd.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener la clase con mayor probabilidad

            print(prediction.tolist())
            # Aquí puedes añadir el código para retornar la predicción
            # Por ejemplo:
            return jsonify({"predicted_class": int(predicted_class), "probabilities": prediction.tolist()}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500


# @app.route('/predict_yolo', methods=['GET', 'POST'])
# def predict_yolo():
#     categories = {
#         0: "*Recomendación: Limpiar, proteger el acero con un inhibidor de corrosión y recubrir con mortero de reparación.",
#         1: "*Recomendación:  Sellar con material elástico o inyectar resina epoxi, según el tipo y movimiento de la grieta",
#         2: "*Recomendación:  Limpiar la zona afectada y rellenar con mortero de reparación de alta adherencia y sin contracción"
#     }
#     if request.method == 'GET':
#         return render_template('prediction_yolo.html')
#     if request.method == 'POST':
#         print('post.......')
#         try:
#             # Verificar si la imagen fue enviada en el request
#             if "image" not in request.files:
#                 return jsonify({"error": "No image uploaded"}), 400
#
#             # Leer la imagen desde el request
#             image_file = request.files["image"]
#             img_path = os.path.join('temp_img_yolo.jpg')
#             image_file.save(img_path)
#             image = Image.open(img_path).convert("RGB")
#
#             imgsz = 348
#
#             # Transformar imagen
#             transform = T.Compose([
#                 T.Resize((imgsz, imgsz)),
#                 T.ToTensor(),  # convierte a float32 y escala [0, 255] -> [0.0, 1.0]
#             ])
#
#             input_tensor = transform(image).unsqueeze(0)  # Añade dimensión batch
#
#             # Inferencia
#             # with torch.no_grad():
#             #     prediction = model_yolo(input_tensor)
#             #prediction = model1.predict(img_array)
#
#             # predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener la clase con mayor probabilidad
#             # print(prediction.tolist())
#             return 200
#             # return jsonify({"prediction":  categories[int(predicted_class)], "probabilities": prediction.tolist()})
#
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request, redirect, jsonify
import datetime
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
import io
import torch
import torchvision.transforms as T

model_rd = tf.keras.models.load_model('model/Model_finalV2.h5')

model_yolo = torch.jit.load('model/best.torchscript')


app = Flask(__name__)
CORS(app)

IMG_SIZE = (328, 328)

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
            img_path = os.path.join('temp_img.jpg')
            image_file.save(img_path)
            img = image.load_img(img_path, target_size=(348, 348))
            # Preprocesar la imagen
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = model_rd.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener la clase con mayor probabilidad

            print(prediction.tolist())

            return jsonify({"prediction":  categories[int(predicted_class)], "probabilities": prediction.tolist()})

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/predict_yolo', methods=['GET', 'POST'])
def predict_yolo():
    categories = {
        0: "*Recomendación: Limpiar, proteger el acero con un inhibidor de corrosión y recubrir con mortero de reparación.",
        1: "*Recomendación:  Sellar con material elástico o inyectar resina epoxi, según el tipo y movimiento de la grieta",
        2: "*Recomendación:  Limpiar la zona afectada y rellenar con mortero de reparación de alta adherencia y sin contracción"
    }
    if request.method == 'GET':
        return render_template('prediction_yolo.html')
    if request.method == 'POST':
        print('post.......')
        try:
            # Verificar si la imagen fue enviada en el request
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded"}), 400

            # Leer la imagen desde el request
            image_file = request.files["image"]
            img_path = os.path.join('temp_img_yolo.jpg')
            image_file.save(img_path)
            image = Image.open(img_path).convert("RGB")

            imgsz = 348

            # Transformar imagen
            transform = T.Compose([
                T.Resize((imgsz, imgsz)),
                T.ToTensor(),  # convierte a float32 y escala [0, 255] -> [0.0, 1.0]
            ])

            input_tensor = transform(image).unsqueeze(0)  # Añade dimensión batch

            # Inferencia
            with torch.no_grad():
                prediction = model_yolo(input_tensor)
            #prediction = model1.predict(img_array)

            predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener la clase con mayor probabilidad
            print(prediction.tolist())

            return jsonify({"prediction":  categories[int(predicted_class)], "probabilities": prediction.tolist()})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
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

model = tf.keras.models.load_model('model/Model_finalV2.h5')

app = Flask(__name__)
CORS(app)

IMG_SIZE = (328, 328)

@app.route("/")
def hello_world():
    return "Hello, World from Flask!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    categories = {
        0: "*Recomendación: Limpiar, proteger el acero con un inhibidor de corrosión y recubrir con mortero de reparación.",
        1: "*Recomendación:  Sellar con material elástico o inyectar resina epoxi, según el tipo y movimiento de la grieta",
        2: "*Recomendación:  Limpiar la zona afectada y rellenar con mortero de reparación de alta adherencia y sin contracción"
    }
    if request.method == 'GET':
        return render_template('prediction.html')
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

            # Hacer la predicción
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener la clase con mayor probabilidad

            print(prediction.tolist())

            return jsonify({"prediction":  categories[int(predicted_class)], "probabilities": prediction.tolist()})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
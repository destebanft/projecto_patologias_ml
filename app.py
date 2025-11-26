from flask import Flask, render_template, request, redirect, jsonify
import datetime
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
import keras


model_rd = keras.models.load_model(
    'model/resnet_fin_tunning_sf_model.keras'
)

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "Hello, World from Flask!"


@app.route('/predict_rd', methods=['GET', 'POST'])
def predict_rd():
    if request.method == 'GET':
        return render_template('prediction_rd.html')
    if request.method == 'POST':

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

            prediction = model_rd.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            probs = tf.nn.softmax(prediction).numpy().tolist()

            return jsonify({"predicted_class": int(predicted_class), "probabilities": probs}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from flask import Flask, request, jsonify,render_template
from werkzeug.utils import secure_filename
import json
from flask_cors import CORS
import warnings
import requests

app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore", category=Warning, module="werkzeug")
app.config['ALLOWED_EXTENSIONS'] = {'apk', 'ipa'}


def predict_class(input_image_path, model_path='model.h5', class_names_path='classes.json'):
    loaded_model = keras.models.load_model(model_path)

    with open(class_names_path, 'r') as json_file:
        class_names = json.load(json_file)

    input_image = load_img(input_image_path, target_size=(224, 224))
    input_image = img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0

    predictions = loaded_model.predict(input_image)

    predicted_class_index = np.argmax(predictions)

    predicted_class_name = class_names[str(predicted_class_index)]

    return predicted_class_name


@app.route('/', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'Image file not provided'})
        image = request.files['image']

        filename = secure_filename(image.filename)
        image_path = os.path.join("uploads", filename)
        image.save(image_path)

        predicted_class = predict_class(image_path)

        return jsonify({'predicted_class': predicted_class})
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

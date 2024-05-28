# Import the necessary libraries
import os
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import Flask Web Service
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Define a flask app
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Ensure you have the correct path to your model
MODEL_PATH = 'models/model_CNN_final_epoch40.h5'

# Load your trained model
new_model = load_model(MODEL_PATH)

# Class
dic = {0 : 'bercak kering',
       1 : 'daun sehat', 
       2 : 'embun tepung', 
       3 : 'tenggorok daun'}

# Class Image Paths
class_images = {
    'bercak kering': 'class/bercak_kering/bercak_kering1.jpg',
    'daun sehat': 'class/daun_sehat/daun_sehat1.jpg',
    'embun tepung': 'class/embun_tepung/embun_tepung1.jpg',
    'tenggorok daun': 'class/tenggorok_daun/tenggorok_daun1.jpg'
}

# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Predict Label
def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = new_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return dic[predicted_class[0]]

# Predict Class
def predict_class(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = new_model.predict(img_array)
    return np.round(prediction[0] * 100)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return 'Tomato Leaf Prediction!'

@app.route('/api/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    f = request.files['file']

    if f.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    try:
        predict = predict_label(file_path)
        prediction = predict_class(file_path)
        os.remove(file_path)  # Remove the file after prediction

        if np.all(prediction < 80):
            return jsonify({'PredictionLabel': 'Daun penyakit tidak ditemukan', 'PredictionClass': prediction})
        else:
            # Get base64 image of the predicted class
            class_image_base64 = get_image_base64(class_images[predict])
            return jsonify({
                'PredictionLabel': predict, 
                'PredictionClass': prediction,
                'ClassImage': class_image_base64
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

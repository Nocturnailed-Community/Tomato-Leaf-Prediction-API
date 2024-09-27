import os
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Path to the model
MODEL_PATH = 'models/model_CNN_final_new_dataset_mobileNetV2_epoch50.h5'

# Attempt to load the model, catch any errors
try:
    new_model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Error loading the model: {e}")
    new_model = None

# Dictionary for class labels
dic = {
    0: 'bercak kering',
    1: 'busuk daun',
    2: 'daun sehat',
    3: 'embun tepung',
    4: 'pengorok daun'
}

# Class images
class_images = {
    'bercak kering': 'class/bercak_kering/bercak_kering1.jpg',
    'busuk daun': 'class/busuk_daun/busuk_daun1.jpg',
    'daun sehat': 'class/daun_sehat/daun_sehat1.jpg',
    'embun tepung': 'class/embun_tepung/embun_tepung1.jpg',
    'pengorok daun': 'class/tenggorok_daun/tenggorok_daun1.jpg',
}

def get_image_base64(image_path):
    """ Convert an image to base64. """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def predict_label(img_path):
    """ Predict the label of the image. """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = new_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        return dic[predicted_class[0]]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def predict_class(img_path):
    """ Predict the class probabilities of the image. """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = new_model.predict(img_array)
        return np.round(prediction[0] * 100).tolist()
    except Exception as e:
        print(f"Error during class prediction: {e}")
        return None

def predict_top_2_classes(img_path):
    """ Predict the top 2 class probabilities of the image. """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = new_model.predict(img_array)[0] * 100  # Get raw predictions and scale to percentage
        
        # Get indices of top 2 predictions
        top_2_indices = np.argsort(prediction)[-2:][::-1]
        top_2_predictions = [(dic[i], round(prediction[i], 2)) for i in top_2_indices]
        
        return top_2_predictions
    except Exception as e:
        print(f"Error during top 2 class prediction: {e}")
        return None
    
@app.route('/', methods=['GET'])
def index():
    """ Test route for API. """
    return jsonify({'message': 'Tomato Leaf Prediction!'})

@app.route('/api/predict', methods=['POST'])
def upload():
    """ Handle file upload and make predictions. """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    # Save the uploaded file
    basepath = os.path.dirname(__file__)
    filename = secure_filename(f.filename)
    file_path = os.path.join(basepath, 'uploads', filename.lower())
    try:
        f.save(file_path)
    except Exception as e:
        return jsonify({'error': f"File saving failed: {e}"}), 500

    try:
        predict = predict_label(file_path)
        prediction = predict_class(file_path)
        
        # Check if prediction is None before proceeding
        # if prediction is None:
        #     raise Exception("Prediction failed or returned no result.")

        # Delete the uploaded file after prediction
        os.remove(file_path)

        # Handle prediction results
        if np.all(np.array(prediction) < 85):
            return jsonify({
                'PredictionLabel': 'Daun penyakit tidak ditemukan', 
                'PredictionClass': prediction,
                'ExtensionImage': 'null',
                'ClassImage': 'null'
            })
        else:
            class_image_base64 = get_image_base64(class_images[predict])
            if class_image_base64 is None:
                raise Exception("Class image not found or could not be encoded.")
                
            return jsonify({
                'PredictionLabel': predict,
                'PredictionClass': prediction,
                'ExtensionImage': 'jpg',
                'ClassImage': class_image_base64
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-top-class', methods=['POST'])
def upload():
    """ Handle file upload and make predictions. """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    # Save the uploaded file
    basepath = os.path.dirname(__file__)
    filename = secure_filename(f.filename)
    file_path = os.path.join(basepath, 'uploads', filename.lower())
    try:
        f.save(file_path)
    except Exception as e:
        return jsonify({'error': f"File saving failed: {e}"}), 500

    try:
        predict = predict_label(file_path)
        top_2_predictions = predict_top_2_classes(file_path)
        
        # Check if prediction is None before proceeding
        if top_2_predictions is None:
            raise Exception("Prediction failed or returned no result.")

        # Delete the uploaded file after prediction
        os.remove(file_path)

        # Handle prediction results
        top_prediction_label, top_prediction_percentage = top_2_predictions[0]
        second_prediction_label, second_prediction_percentage = top_2_predictions[1]

        # Retrieve class images for the top 2 predictions
        top_class_image_base64 = get_image_base64(class_images.get(top_prediction_label, None))
        second_class_image_base64 = get_image_base64(class_images.get(second_prediction_label, None))

        # Check if prediction is below 85% for both classes
        if top_prediction_percentage < 85:
            return jsonify({
                'PredictionLabel': 'Daun penyakit tidak ditemukan', 
                'Top2Predictions': top_2_predictions,
                'Top2ClassImages': ['null', 'null'],
                'ExtensionImage': 'null',
            })
        else:
            return jsonify({
                'PredictionLabel': predict,
                'Top2Predictions': top_2_predictions,
                'Top2ClassImages': [
                    {'label': top_prediction_label, 'image': top_class_image_base64},
                    {'label': second_prediction_label, 'image': second_class_image_base64}
                ],
                'ExtensionImage': 'jpg'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
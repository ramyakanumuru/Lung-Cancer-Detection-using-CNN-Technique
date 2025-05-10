from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)


model = load_model('lung_cancer_model.h5')


categories = {
    'BenginCases': 'Benign cases',
    'MalignantCases': 'Malignant cases',
    'Normal': 'Normal',
    'Adenocarcinoma': 'Adenocarcinoma',
    'LargeCellCarcinoma': 'Large cell carcinoma',
    'SquamousCellCarcinoma': 'Squamous cell carcinoma'
}

def preprocess_image(image, img_size=256):
    """Preprocess the uploaded image for prediction."""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img.reshape(-1, img_size, img_size, 1) / 255.0
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    img = preprocess_image(file)

   
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = list(categories.keys())[predicted_class_index]
    
   
    if predicted_class_label == 'Normal':
        result = "No lung cancer detected."
    else:
        result = f"Lung cancer detected: {categories[predicted_class_label]}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

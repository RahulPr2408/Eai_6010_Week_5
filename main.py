import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU attempts

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from waitress import serve

app = Flask(__name__)

# App Engine compatible upload folder
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('model/mnist_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image_array = 1 - (np.array(image) / 255.0)
    return np.expand_dims(image_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction = model.predict(preprocess_image(Image.open(filepath)))
            return render_template('result.html', 
                               filename=filename,
                               prediction=np.argmax(prediction),
                               confidence=np.max(prediction)*100)
    
    return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
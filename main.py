import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from waitress import serve
from io import BytesIO
import base64

app = Flask(__name__)
model = load_model('model/mnist_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image_array = 1 - (np.array(image) / 255.0)
    return np.expand_dims(image_array, axis=0)

def resize_image(img, max_size=400):
    width, height = img.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width*ratio), int(height*ratio))
        return img.resize(new_size, Image.LANCZOS)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            img = resize_image(Image.open(file.stream))
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            prediction = model.predict(preprocess_image(img))
            return render_template('result.html',
                               image_data=img_str,
                               prediction=np.argmax(prediction),
                               confidence=np.max(prediction)*100)
    
    return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
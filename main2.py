from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import dlib
import numpy as np
from PIL import Image
import json
import os


app = Flask(__name__)

@app.route('/api/driver/pre-processing2', methods=["POST"])
def preProcessing2():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({'error': 'No file found in the request'})
        file = request.files['file']
        # print(file.content_type)
        # print(file)
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Save the uploaded image file
        filename = secure_filename(file.filename)
        file.save(filename)

        # Load the input image
        image = cv2.imread(filename)

        # Resize the image
        resized_img = cv2.resize(image, (224, 224))
        # Normalize the image
        normalized_img = resized_img / 255.0
        # Add a batch dimension
        input_img = np.expand_dims(normalized_img, axis=0)

        os.remove(filename)
        left_eye_list = input_img.tolist()
        return jsonify({"msg" : left_eye_list})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

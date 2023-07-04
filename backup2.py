from flask import Flask, request, jsonify , send_file
from werkzeug.utils import secure_filename
import cv2
import dlib
import numpy as np
from PIL import Image
import json

def preprocess2(img):

    img_cv = cv2.imread(img)
    # Resize the image
    resized_img = cv2.resize(img_cv, (224, 224))
    # Normalize the image
    normalized_img = resized_img / 255.0
    # Add a batch dimension
    input_img = np.expand_dims(normalized_img, axis=0)
    return input_img

app = Flask(__name__)

@app.route('/api/driver/pre-processinggg', methods=["POST"])
def preProcessing2():
    if request.method == "POST":
        image = request.files['image']
        sayed = preprocess2("./5.jpg")
        # sayed.save("./uploaded/alo12.jpg")
        left_eye_list = sayed.tolist()
    return jsonify({"msg" : left_eye_list})
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)

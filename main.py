# imports
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2 , dlib , numpy as np


app = Flask(__name__)

# Distraction route
@app.route('/api/driver/distraction-pre-processing', methods=["POST"])
def distractionPreProcessing():
    try:
        if request.method == "POST":
            if 'file' not in request.files:
                return jsonify({'msg': 'No file found in the request'})
            
            file = request.files['file']

            if file.filename == '':
                return jsonify({'msg': 'No file selected'})
            
            # Save the uploaded image file
            filename = secure_filename(file.filename)

            # Load the input image
            image = cv2.imread(filename)

            # Resize the image
            resized_img = cv2.resize(image, (224, 224))

            # Normalize the image
            normalized_img = resized_img / 255.0
            
            # Add a batch dimension
            input_img = np.expand_dims(normalized_img, axis=0)

            img = input_img.tolist()
            return jsonify({"msg" : img})
    except Exception as e:
        return jsonify({'msg': str(e)})

#--------------------------------------------------------------------------------

# Drowsiness route
@app.route('/api/driver/drowsing-pre-processing', methods=["POST"])
def drowsingPreProcessing():
    try:
        if request.method == "POST":
            if 'file' not in request.files:
                return jsonify({'msg': 'No file found in the request'})

            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'msg': 'No file selected'})

            # Save the uploaded image file
            filename = secure_filename(file.filename)

            # Load the input image
            image = cv2.imread(filename)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load the face detector from dlib
            detector = dlib.get_frontal_face_detector()

            # Load the facial landmark predictor from dlib
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
            # Detect faces in the grayscale image
            faces = detector(gray)

            # Create empty lists to store the eye images
            left_eyes = []
            right_eyes = []

            # Loop over the detected faces
            for face in faces:
                # Get the facial landmarks for the face
                landmarks = predictor(gray, face)

                # Get the coordinates of the left eye landmarks
                left_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

                # Get the coordinates of the right eye landmarks
                right_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                # Compute the dimensions of the left eye
                left_x_begin = min(left_eye_coords, key=lambda x: x[0])[0]-10
                left_x_end = max(left_eye_coords, key=lambda x: x[0])[0]+10
                left_y_begin = min(left_eye_coords, key=lambda x: x[1])[1]-10
                left_y_end = max(left_eye_coords, key=lambda x: x[1])[1]+10

                # Compute the dimensions of the right eye
                right_x_begin = min(right_eye_coords, key=lambda x: x[0])[0]-10
                right_x_end = max(right_eye_coords, key=lambda x: x[0])[0]+10
                right_y_begin = min(right_eye_coords, key=lambda x: x[1])[1]-10
                right_y_end = max(right_eye_coords, key=lambda x: x[1])[1]+10

                # Extract the region of interest (ROI) for each eye
                left_eye_roi = image[left_y_begin:left_y_end , left_x_begin:left_x_end]
                right_eye_roi = image[right_y_begin:right_y_end , right_x_begin:right_x_end]

                # preprocess 
                left_eye_img = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2RGB)
                left_eye_img = cv2.resize(left_eye_img, (64, 64))  # Resize to match model input shape
                left_eye_img = left_eye_img / 255.0  # Normalize pixel values
                left_eye_img = np.expand_dims(left_eye_img, axis=0)  # Add batch dimension

                right_eye_img = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2RGB)
                right_eye_img = cv2.resize(right_eye_img, (64, 64))  # Resize to match model input shape
                right_eye_img = right_eye_img / 255.0  # Normalize pixel values
                right_eye_img = np.expand_dims(right_eye_img, axis=0)  # Add batch dimension

                # Append the eye images to the lists
                left_eyes.append(left_eye_img)
                right_eyes.append(right_eye_img)

            # Return the extracted eye images
            left_eye_list = left_eyes[0].tolist()
            return jsonify({"msg" : left_eye_list})
    except Exception as e:
        return jsonify({'msg': str(e)})



if __name__ == "__main__":
    app.run(debug=True, port=5000)

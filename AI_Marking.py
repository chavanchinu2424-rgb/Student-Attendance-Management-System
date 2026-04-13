from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Create a temp folder to process images
if not os.path.exists("temp"):
    os.makedirs("temp")

def save_base64_img(bs64, filename):
    if "base64," in bs64: bs64 = bs64.split("base64,")[1]
    with open(filename, "wb") as f:
        f.write(base64.b64decode(bs64))

@app.route('/api/verify-face', methods=['POST'])
def verify():
    try:
        data = request.json
        stored_path = "temp/stored.jpg"
        live_path = "temp/live.jpg"

        save_base64_img(data['stored_image'], stored_path)
        save_base64_img(data['live_image'], live_path)

        # DeepFace is highly precise. 
        # model_name="VGG-Face" is the industry standard for profile matching.
        result = DeepFace.verify(
            img1_path = stored_path, 
            img2_path = live_path, 
            model_name = "VGG-Face",
            enforce_detection = True
        )

        return jsonify({"match": bool(result["verified"])})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"match": False, "error": "No face detected in frame"})

if __name__ == '__main__':
    app.run(port=8000)
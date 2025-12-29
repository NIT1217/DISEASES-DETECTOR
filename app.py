from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


model = tf.keras.models.load_model('model.h5') 

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predicted():
    if 'xray_report' not in request.files:
        return "No file uploaded"

    file = request.files['xray_report']
    if file.filename == '':
        return "No file selected"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # --- Preprocessing (Match your Notebook settings) ---
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 
   
    prediction = model.predict(img_array)
    classes = ['Normal', 'Pneumonia', 'Tuberculosis']
    pred_label = classes[np.argmax(prediction)]

    return render_template(
        "index2.html",
        response=f"The predicted disease in your X-ray report is: {pred_label}"
    )

if __name__ == "__main__":
    app.run(debug=True)

    
    



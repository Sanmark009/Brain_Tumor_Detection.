from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize the Flask application
app = Flask(__name__)

# Load the trained brain tumor detection model
model = load_model(r'brain_tumor_models.h5')  # Replace with your model path

# Load class indices from JSON (generated during training)
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reconstruct the class labels list in the correct index order
TUMOR_CLASSES = [None] * len(class_indices)
for label, index in class_indices.items():
    TUMOR_CLASSES[index] = label.replace('_', ' ')  # Optional: clean label names

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image size (should match training input size)
IMG_SIZE = (224, 224)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to create the pie chart for tumor spreading percentage based on the tumor type
def create_spreading_pie_chart(predicted_class):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set different tumor spreading percentages based on tumor types
    if predicted_class == 'Glioma Tumor':
        spreading_data = [70, 30]  # Example: 70% tumor spread, 30% non-tumor area
    elif predicted_class == 'Meningioma Tumor':
        spreading_data = [60, 40]  # Example: 60% tumor spread, 40% non-tumor area
    elif predicted_class == 'Pituitary Tumor':
        spreading_data = [50, 50]  # Example: 50% tumor spread, 50% non-tumor area
    elif predicted_class == 'No Tumor':
        spreading_data = [0, 100]  # No tumor, 0% spread

    # Create pie chart
    ax.pie(spreading_data, labels=['Tumor Spread', 'Non-Tumor Area'], autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#cccccc'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Convert to base64 image to render in the browser
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    error_message = None
    spreading_pie_chart_url = None
    tumor_description = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = 'No file part'
            return render_template('index.html', error_message=error_message)

        file = request.files['file']

        if file.filename == '':
            error_message = 'No selected file'
            return render_template('index.html', error_message=error_message)

        if not allowed_file(file.filename):
            error_message = 'Only image files are allowed (PNG, JPG, JPEG)'
            return render_template('index.html', error_message=error_message)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = TUMOR_CLASSES[predicted_class_idx]

        # Tumor Description (you can customize this based on the class)
        tumor_description = f"This is a {predicted_class}. The model detected the tumor based on the provided image."

        # Create the tumor spreading pie chart based on the predicted class
        spreading_pie_chart_url = create_spreading_pie_chart(predicted_class)

    return render_template('index.html', prediction=prediction, img_path=img_path, 
                           error_message=error_message, tumor_description=tumor_description,
                           spreading_pie_chart_url=spreading_pie_chart_url)

if __name__ == '__main__':
    app.run(debug=True)

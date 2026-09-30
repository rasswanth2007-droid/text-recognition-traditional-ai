from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from traditional_ocr import TraditionalOCR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OCR
ocr = TraditionalOCR()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run recognition
        try:
            result_text = ocr.recognize_image(filepath)
            if result_text is None:
                return jsonify({'error': 'Failed to process image'}), 500
            
            return jsonify({'text': result_text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
from vit_image_analysis import ImageAnalyzer

app = Flask(__name__)
analyzer = ImageAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save the image temporarily
        temp_path = 'temp_image.jpg'
        image.save(temp_path)
        
        # Analyze the image
        predicted_label, confidence = analyzer.analyze_image(temp_path)
        
        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence
        })

if __name__ == '__main__':
    app.run(debug=True)
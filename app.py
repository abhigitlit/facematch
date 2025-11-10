import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis
import base64

app = Flask(__name__)
CORS(app)

# Initialize FaceAnalysis
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def decode_base64_image(base64_string):
    """
    Decode base64 image string to OpenCV image format
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        img_array = np.frombuffer(img_bytes, np.uint8)
        
        # Decode to OpenCV image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/api/compare', methods=['POST'])
def compare_faces():
    """
    Compare two faces from uploaded images
    """
    try:
        data = request.json
        
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Both images are required'}), 400
        
        # Decode images
        img1 = decode_base64_image(data['image1'])
        img2 = decode_base64_image(data['image2'])
        
        if img1 is None or img2 is None:
            return jsonify({'error': 'Could not decode one or both images'}), 400
        
        # Detect faces
        faces1 = face_app.get(img1)
        faces2 = face_app.get(img2)
        
        if not faces1:
            return jsonify({'error': 'No face detected in first image'}), 400
        
        if not faces2:
            return jsonify({'error': 'No face detected in second image'}), 400
        
        # Get embeddings
        feat1 = faces1[0].embedding
        feat2 = faces2[0].embedding
        
        # Calculate similarity (cosine similarity)
        similarity = float(np.dot(feat1, feat2))
        
        # Determine if it's a match (threshold: 0.5)
        is_match = similarity > 0.5
        
        return jsonify({
            'similarity': round(similarity, 4),
            'isMatch': is_match,
            'threshold': 0.5,
            'message': 'Match found! Likely the same person.' if is_match else 'No match. Likely different people.'
        })
        
    except Exception as e:
        print(f"Error in compare_faces: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'ok', 'message': 'Face comparison API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
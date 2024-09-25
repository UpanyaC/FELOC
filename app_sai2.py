from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import base64
import cv2
import numpy as np

app = Flask(__name__)

# Store user data
user_data = {
    "expressions": [],  # List to store multiple expressions
    "facial_features": []  # List to store corresponding facial features
}

@app.route('/')
def index():
    return render_template('user2.html')

@app.route('/save_expression_page')
def save_expression_page():
    return render_template('user_save2.html')

@app.route('/login_page')
def login_page():
    return render_template('user_login2.html')

@app.route('/save_expression_video', methods=['POST'])
def save_expression_video():
    try:
        data = request.json
        video_data = data['video'].split(',')[1]
        video_bytes = base64.b64decode(video_data)
        
        # Save the video file for processing
        with open('saved_expression_video.webm', 'wb') as video_file:
            video_file.write(video_bytes)
        
        # Read the video file using OpenCV
        cap = cv2.VideoCapture('saved_expression_video.webm')
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every nth frame to avoid redundancy (adjust as needed)
            if frame_count % 10 == 0:
                # Debugging: Save each frame for verification
                cv2.imwrite(f"frame_{frame_count}.jpg", frame)
                print(f"Saved frame {frame_count} for debugging.")

                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                embedding = DeepFace.represent(frame, enforce_detection=False)
                
                if 'dominant_emotion' in analysis[0]:
                    user_data['expressions'].append(analysis[0]['dominant_emotion'])
                    user_data['facial_features'].append(embedding[0]['embedding'])
                    print(f"Saved expression: {analysis[0]['dominant_emotion']}")
                    print(f"Saved facial features: {embedding[0]['embedding']}")
        
        cap.release()
        
        # Print all saved expressions and features for verification
        print("All saved expressions:", user_data['expressions'])
        print("All saved facial features:", [len(f) for f in user_data['facial_features']])
        
        return jsonify({'result': 'Video processed and expressions saved successfully!'})
    except Exception as e:
        print(f"Error in save_expression_video: {str(e)}")
        return jsonify({'error': str(e)})


def verify(img):
    try:
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=True)
        embedding = DeepFace.represent(img, enforce_detection=True)
        
        if 'dominant_emotion' in analysis[0] and embedding:
            return analysis[0]['dominant_emotion'], embedding[0]['embedding']
        else:
            return None, None
    except Exception as e:
        print(f"Error in verify: {str(e)}")
        return None, str(e)

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Debugging: Save the image for verification
        cv2.imwrite("login_image.jpg", img)
        print("Saved login image for debugging.")

        captured_emotion, captured_features = verify(img)
        print(f"Captured emotion: {captured_emotion}")
        print(f"Captured features: {captured_features}")
        
        if captured_emotion and captured_features:
            for saved_emotion, saved_features in zip(user_data.get('expressions', []), user_data.get('facial_features', [])):
                distance = np.linalg.norm(np.array(captured_features) - np.array(saved_features))
                print(f"Comparing with saved emotion: {saved_emotion}, distance: {distance}")

                if distance < 0.6:  # Adjust threshold as needed
                    if captured_emotion == saved_emotion:
                        return jsonify({'result': 'Access granted!', 'emotion': captured_emotion})
            
            return jsonify({'result': 'Access denied: Facial features or expressions do not match', 'emotion': captured_emotion})
        else:
            return jsonify({'error': 'Dominant emotion or facial features not found in analysis'})
    except Exception as e:
        print(f"Error in login: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

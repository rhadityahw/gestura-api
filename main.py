import firebase_admin
from flask_cors import CORS
from firebase_admin import credentials, firestore, storage
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertForSequenceClassification, AutoModel,TrainingArguments, Trainer
import torch


import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
CORS(app)

# Text Classification Prep
pretrained = "indolem/indobert-base-uncased"
model_directory = "./gestura_text_model"
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = BertForSequenceClassification.from_pretrained(model_directory, use_safetensors=True)



if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Firebase
cred = credentials.Certificate('sign-language-app-7bccf-firebase-adminsdk-l25bu-2a41ac811d.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'gs://sign-language-app-7bccf.appspot.com/kamus' 
})

db = firestore.client()
bucket = storage.bucket()

# Endpoint to upload video (kata)
@app.route('/upload-kata', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    name = request.form.get('name')
    file = request.files['file']
    if not name:
        return jsonify({"error": "No name provided"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        blob = bucket.blob(filename)
        blob.upload_from_filename(filepath)

        video_url = blob.public_url
        data = {
            'Nama': name,
            'Video': video_url
        }

        db.collection('kamus_kata').add(data)

        return jsonify({"message": "File uploaded successfully", "video_url": video_url}), 201

# Endpoint to retrieve all documents (kata)
@app.route('/videos-kata', methods=['GET'])
def get_videos_kata():
    try:
        videos = []
        docs = db.collection('kamus_kata').stream() 
        for doc in docs:
            video = doc.to_dict()
            video['id'] = doc.id
            videos.append(video)
        return jsonify(videos), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to retrieve a video (kata) by name
@app.route('/videos-kata/<name>', methods=['GET'])
def get_video_kata_by_name(name):
    try:
        params = name.lower()
        videos_ref = db.collection('kamus_kata') 
        query = videos_ref.where('Nama', '==', params).stream()

        results = []
        for doc in query:
            results.append(doc.to_dict())

        if results:
            return jsonify(results), 200
        else:
            return jsonify({"error": "No video found with the given name"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Endpoint to retrieve all documents (huruf)
@app.route('/videos-huruf', methods=['GET'])
def get_videos_huruf():
    try:
        videos = []
        docs = db.collection('kamus_huruf').stream() 
        for doc in docs:
            video = doc.to_dict()
            video['id'] = doc.id
            videos.append(video)
        return jsonify(videos), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Endpoint to retrieve a video (huruf) by name
@app.route('/videos-huruf/<name>', methods=['GET'])
def get_video_huruf_by_name(name):
    try:
        params = name.lower()
        videos_ref = db.collection('kamus_huruf') 
        query = videos_ref.where('Nama', '==', params).stream()

        results = []
        for doc in query:
            results.append(doc.to_dict())

        if results:
            return jsonify(results), 200
        else:
            return jsonify({"error": "No video found with the given name"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/hate-speech/<text_input>', methods=['GET'])
def classify_text(text_input):
    try:
        inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        print(text_input)
        return jsonify({
            "result": prediction.item() == 1
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)

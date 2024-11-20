import os
import tempfile
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import scipy.signal
import onnxruntime as ort
import requests
import openai
import logging
from google.cloud import storage

# Initialize Flask app
app = Flask(__name__)

# Initialize logger with more detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.debug(f"Successfully downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def load_models():
    """Load all required models and scalers"""
    try:
        ensemble_models = []
        ensemble_paths = [
            "onnx_multi_model5a.onnx",
            "onnx_multi_model3a.onnx",
            "onnx_multi_model1a.onnx",
        ]

        bucket_name = "ser_models"
        model_folder = "models"
        scaler_folder = "scalers"
        destination_folder = "/tmp"

        # Set Google credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'

        # Download and load models
        for path in ensemble_paths:
            source_blob_name = f"{model_folder}/{path}"
            destination_file_name = f"{destination_folder}/{path}"
            download_model(bucket_name, source_blob_name, destination_file_name)
            model = ort.InferenceSession(destination_file_name)
            ensemble_models.append(model)

        # Download and load scaler
        scaler_path = "scaler_multi.joblib"
        source_blob_name = f"{scaler_folder}/{scaler_path}"
        destination_file_name = f"{destination_folder}/{scaler_path}"
        download_model(bucket_name, source_blob_name, destination_file_name)
        scaler = joblib.load(destination_file_name)

        # Download and load label encoder
        label_encoder_path = "label_multi.joblib"
        source_blob_name = f"{scaler_folder}/{label_encoder_path}"
        destination_file_name = f"{destination_folder}/{label_encoder_path}"
        download_model(bucket_name, source_blob_name, destination_file_name)
        label_encoder = joblib.load(destination_file_name)

        logger.info("Successfully loaded all models and scalers")
        return ensemble_models, scaler, label_encoder
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Initialize models before starting the app
ensemble_models, scaler, label_encoder = load_models()

def process_recorded_audio(audio_data, sr=16000):
    """Handle recorded audio processing"""
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=16000)
        
        # Apply preprocessing
        audio_data = preprocess_audio(audio_data)
        return audio_data
    except Exception as e:
        logger.error(f"Error processing recorded audio: {str(e)}")
        raise

def preprocess_audio(audio_data):
    """Common preprocessing pipeline"""
    try:
        # Remove DC offset
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply pre-emphasis filter
        audio_data = librosa.effects.preemphasis(audio_data)
        
        # Apply noise reduction
        audio_data = reduce_noise(audio_data)
        
        return audio_data
    except Exception as e:
        logger.error(f"Error in preprocessing audio: {str(e)}")
        raise

def reduce_noise(audio_data):
    """Noise reduction implementation"""
    try:
        # Calculate noise floor from first 10% of signal
        noise_floor = np.mean(np.abs(audio_data[:int(len(audio_data)/10)]))
        
        # Apply soft thresholding
        audio_data = np.where(
            np.abs(audio_data) < noise_floor * 2,
            audio_data * 0.1,
            audio_data
        )
        return audio_data
    except Exception as e:
        logger.error(f"Error reducing noise: {str(e)}")
        raise

def frft(x, alpha):
    """Fractional Fourier Transform implementation"""
    try:
        N = len(x)
        t = np.arange(N)
        kernel = np.exp(-1j * np.pi * alpha * t**2 / N)
        return scipy.signal.fftconvolve(x, kernel, mode='same')
    except Exception as e:
        logger.error(f"Error in FRFT calculation: {str(e)}")
        raise

def extract_features(data, sample_rate):
    """Extract audio features"""
    try:
        # Extract MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
        delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
        acceleration_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
        
        # Extract mel spectrogram
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

        # Calculate FRFT features
        alpha_values = np.linspace(0.1, 0.9, 9)
        frft_features = np.array([])
        for alpha in alpha_values:
            frft_result = frft(data, alpha)
            frft_features = np.hstack((frft_features, np.mean(frft_result.real, axis=0)))

        # Extract spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)

        # Combine all features
        return np.hstack((mfcc, delta_mfcc, acceleration_mfcc, mel_spectrogram, frft_features, spectral_centroid))
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def normalize_audio(audio):
    """Normalize audio data"""
    try:
        audio = audio.astype(np.float32)
        return librosa.util.normalize(audio)
    except Exception as e:
        logger.error(f"Error normalizing audio: {str(e)}")
        raise

def trim_silences(data, sr, top_db=35):
    """Remove silence from audio"""
    try:
        return librosa.effects.trim(data, top_db=top_db)[0]
    except Exception as e:
        logger.error(f"Error trimming silences: {str(e)}")
        raise

def prepare_audio_length(data, sr, target_duration=3):
    """Prepare audio to consistent length"""
    try:
        target_length = sr * target_duration
        current_length = len(data)
        
        if current_length < target_length:
            if len(data) < sr:  # If less than 1 second
                data = librosa.effects.time_stretch(data, rate=len(data)/(sr*target_duration))
                if len(data) > target_length:
                    return data[:target_length]
                elif len(data) < target_length:
                    return np.pad(data, (0, target_length - len(data)), mode='wrap')
            else:
                repetitions = int(np.ceil(target_length / current_length))
                data_repeated = np.tile(data, repetitions)
                return data_repeated[:target_length]
        
        return data
    except Exception as e:
        logger.error(f"Error preparing audio length: {str(e)}")
        raise

def generate_windows(data, window_size, hop_size, sr):
    """Generate sliding windows from audio data"""
    try:
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        windows = []
        
        for i in range(0, len(data) - window_samples + 1, hop_samples):
            window = data[i:i + window_samples]
            windows.append(window)
        
        return windows
    except Exception as e:
        logger.error(f"Error generating windows: {str(e)}")
        raise

def predict_with_onnx_model(model, features):
    """Make predictions using ONNX model"""
    try:
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        features = features.astype(np.float32).reshape(1, 153, 1, 1)
        return model.run([label_name], {input_name: features})[0]
    except Exception as e:
        logger.error(f"Error predicting with ONNX model: {str(e)}")
        raise

def predict_emotion(audio_file, scaler, window_size=3.0, hop_size=0.75):
    """Predict emotions from audio"""
    try:
        # Load and preprocess audio
        if isinstance(audio_file, str):
            data, sr = librosa.load(audio_file, sr=16000)
        else:
            data = audio_file
            sr = 16000
        
        # Apply preprocessing pipeline
        data = process_recorded_audio(data, sr)
        data = trim_silences(data, sr)
        data = normalize_audio(data)
        data = prepare_audio_length(data, sr, target_duration=3)

        # Generate windows
        windows = generate_windows(data, window_size, hop_size, sr)
        if not windows:
            return {label: "0.00%" for label in label_encoder.classes_}

        # Process each window
        emotion_probs = np.zeros(len(label_encoder.classes_))
        window_weights = np.linspace(1.0, 2.0, len(windows))

        for idx, window in enumerate(windows):
            features = extract_features(window, sr)
            if len(features) != 153:
                raise ValueError(f"Expected 153 features, got {len(features)}")

            features_scaled = scaler.transform(features.reshape(1, -1))
            window_probs = np.mean([predict_with_onnx_model(model, features_scaled) 
                                  for model in ensemble_models], axis=0).squeeze()
            emotion_probs += window_probs * window_weights[idx]

        # Normalize probabilities
        emotion_probs /= np.sum(window_weights)

        return {label: f"{prob * 100:.2f}%" 
                for label, prob in zip(label_encoder.classes_, emotion_probs)}
    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        raise

def transcribe_audio(audio_file_path):
    """Transcribe audio to text"""
    try:
        audio = AudioSegment.from_file(audio_file_path)
        wav_file_path = "/tmp/temp_transcription.wav"
        audio.export(wav_file_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return "[unrecognized]"
            except sr.RequestError as e:
                return f"Transcription error: {e}"
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise
    finally:
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

def get_llm_interpretation(emotional_results, transcription):
    """Get LLM interpretation of results"""
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        prompt = f"""
            Expert audio emotion analysis:
            Emotional results: {emotional_results}
            Transcript: {transcription}
            Provide succinct interpretation of emotional content.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error getting LLM interpretation: {str(e)}")
        raise

def process_audio_file(audio_file):
    """Process audio file and return results"""
    try:
        prediction = predict_emotion(audio_file, scaler)
        transcription = transcribe_audio(audio_file)
        llm_interpretation = get_llm_interpretation(prediction, transcription)
        return prediction, transcription, llm_interpretation
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio endpoint"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        
        # Handle different audio formats
        if audio_file.filename.endswith('.webm'):
            # Convert webm to wav
            audio = AudioSegment.from_file(audio_file, format="webm")
            audio_file_path = '/tmp/recorded_audio.wav'
            audio.export(audio_file_path, format="wav")
        else:
            audio_file_path = '/tmp/' + audio_file.filename
            audio_file.save(audio_file_path)

        # Process audio
        predictions, transcription, llm_interpretation = process_audio_file(audio_file_path)

        # Cleanup
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        return jsonify({
            "Emotion Probabilities": predictions,
            "Transcription": transcription,
            "LLM Interpretation": llm_interpretation,
        })

    except Exception as e:
        logger.error(f"Error in process_audio endpoint: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

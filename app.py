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

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.debug(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def load_models():
    ensemble_models = []
    ensemble_paths = [
        "onnx_multi_model5a.onnx",
        #"onnx_multi_model4a.onnx",
        "onnx_multi_model3a.onnx",
        #"onnx_multi_model2a.onnx",
        "onnx_multi_model1a.onnx",
    ]

    bucket_name = "ser_models"
    model_folder = "models"
    scaler_folder = "scalers"
    destination_folder = "/tmp"  # Local folder where you want to save the models

    # Set the environment variable for application credentials
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

    return ensemble_models, scaler, label_encoder

# Initialize models before starting the app
ensemble_models, scaler, label_encoder = load_models()

def frft(x, alpha):
    """Fractional Fourier Transform implementation"""
    N = len(x)
    t = np.arange(N)
    kernel = np.exp(-1j * np.pi * alpha * t**2 / N)
    return scipy.signal.fftconvolve(x, kernel, mode='same')

def extract_features(data, sample_rate):
    """Extract audio features"""
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
        delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
        acceleration_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

        alpha_values = np.linspace(0.1, 0.9, 9)
        frft_features = np.array([])

        for alpha in alpha_values:
            frft_result = frft(data, alpha)
            frft_features = np.hstack((frft_features, np.mean(frft_result.real, axis=0)))

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)

        return np.hstack((mfcc, delta_mfcc, acceleration_mfcc, mel_spectrogram, frft_features, spectral_centroid))
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def normalize_audio(audio):
    """Normalize audio data"""
    try:
        original_max = np.abs(audio).max()
        audio = audio.astype(np.float32)
        normalized_audio = np.clip(audio / original_max, -1.0, 1.0)
        return normalized_audio
    except Exception as e:
        logger.error(f"Error normalizing audio: {str(e)}")
        raise

def trim_silences(data, sr, top_db=35):
    """Remove silence from audio"""
    try:
        trimmed_data, _ = librosa.effects.trim(data, top_db=top_db)
        return trimmed_data
    except Exception as e:
        logger.error(f"Error trimming silences: {str(e)}")
        raise

def generate_windows(data, window_size, hop_size, sr):
    """Generate sliding windows from audio data"""
    try:
        num_samples = len(data)
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        windows = []
        for i in range(0, num_samples - window_samples + 1, hop_samples):
            window = data[i:i + window_samples]
            windows.append(window)

        return windows
    except Exception as e:
        logger.error(f"Error generating windows: {str(e)}")
        raise

def predict_with_onnx_model(model, features):
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    features = features.astype(np.float32).reshape(1, 153, 1, 1)
    prediction = model.run([label_name], {input_name: features})
    return prediction[0]

def predict_emotion(audio_file, scaler, window_size=3.0, hop_size=0.5):
    """
    Enhanced emotion prediction with robust window handling and confidence scores
    """
    try:
        # Load audio
        data, sr = librosa.load(audio_file, sr=16000)
        
        # Calculate window and hop sizes in samples
        window_length = int(window_size * sr)
        hop_length = int(hop_size * sr)
        
        # Split audio into overlapping windows
        windows = []
        for i in range(0, len(data) - window_length + 1, hop_length):
            window = data[i:i + window_length]
            if len(window) == window_length:
                try:
                    # Process each window with exact training sequence
                    trimmed_window, _ = librosa.effects.trim(window)
                    normalized_window = normalize_audio(trimmed_window)
                    
                    # Ensure 3 seconds duration
                    if len(normalized_window) > window_length:
                        normalized_window = normalized_window[:window_length]
                    else:
                        pad_length = window_length - len(normalized_window)
                        normalized_window = np.pad(normalized_window, (0, pad_length), 'constant')
                    
                    windows.append(normalized_window)
                except Exception as trim_error:
                    logger.warning(f"Skipping window due to: {str(trim_error)}")
                    continue

        # Handle case when no valid windows are found
        if not windows:
            try:
                trimmed_data, _ = librosa.effects.trim(data)
                normalized_data = normalize_audio(trimmed_data)
                if len(normalized_data) > window_length:
                    normalized_data = normalized_data[:window_length]
                else:
                    pad_length = window_length - len(normalized_data)
                    normalized_data = np.pad(normalized_data, (0, pad_length), 'constant')
                windows = [normalized_data]
            except Exception as e:
                raise ValueError(f"Error processing full audio: {str(e)}")

        # Process windows and get predictions
        window_predictions = []
        for window in windows:
            features = extract_features(window, sr)
            
            if len(features) != 153:  # Ensure correct feature dimension
                logger.warning(f"Unexpected feature length: {len(features)}, skipping window")
                continue
                
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get predictions from all models in ensemble
            window_ensemble_preds = []
            for model in ensemble_models:
                pred = predict_with_onnx_model(model, features_scaled)
                window_ensemble_preds.append(pred)
            
            window_pred = np.mean(window_ensemble_preds, axis=0)
            window_predictions.append(window_pred)

        if not window_predictions:
            raise ValueError("No valid predictions obtained from any window")

        # Calculate final predictions with confidence scores
        final_prediction = np.mean(window_predictions, axis=0)
        ensemble_std = np.std([pred for preds in window_predictions for pred in preds], axis=0)
        window_std = np.std(window_predictions, axis=0)
        confidence_scores = 100 * (1 - (ensemble_std + window_std) / 2)

        # Format results with confidence scores
        results = {}
        for label, prob, conf in zip(label_encoder.classes_, final_prediction.squeeze(), confidence_scores):
            if prob > 0.05:  # Only include emotions with >5% probability
                results[label] = {
                    'probability': f"{prob * 100:.2f}%",
                    'confidence': f"{conf:.2f}%"
                }

        # Sort by probability
        return dict(sorted(results.items(),
                         key=lambda x: float(x[1]['probability'].strip('%')),
                         reverse=True))

    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        raise

def transcribe_audio(audio_file_path):
    """Transcribe audio to text"""
    try:
        audio = AudioSegment.from_file(audio_file_path)
        wav_file_path = "/tmp/uploaded_audio.wav"
        audio.export(wav_file_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_data)
                return transcription
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
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"""
            You are an expert in audio emotion recognition and analysis. Given the following information:
    
                Audio data details:
                - Emotional recognition results: {emotional_results}
                - Transcript: {transcription}
    
                Your task is to provide a very succinct and insightful interpretation of the emotional content captured in the audio data, considering both the emotion recognition results and the transcript.
    
                In your response, please:
    
                <thinking>
                - Summarize the key emotions detected by the model and their relative strengths.
                - Discuss how the emotions expressed in the transcript align with or differ from the model's predictions.
                - Analyze any notable patterns or trends in the emotional content, especially changes in emotional state over time, differences between speakers, or contextual factors influencing the emotions.
                - Highlight the most salient and informative aspects of the emotional data that would be valuable for understanding the overall emotional experience captured in the audio.
                </thinking>
    
                <result>
                Based on the provided information, your succinct and insightful interpretation of the emotional content in the audio data is:
                </result>
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

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


def process_audio_file(audio_file):
    """Process audio file and return results with confidence scores"""
    try:
        prediction = predict_emotion(audio_file, scaler)
        transcription = transcribe_audio(audio_file)
        llm_interpretation = get_llm_interpretation(prediction, transcription)
        
        return {
            "Emotion Analysis": prediction,  # Now includes probabilities and confidence scores
            "Transcription": transcription,
            "LLM Interpretation": llm_interpretation
        }
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    temp_dir = tempfile.mkdtemp()
    temp_wav = os.path.join(temp_dir, 'audio.wav')
    
    try:
        print(f"Processing file: {audio_file.filename}")
        
        # Check if it's a recorded blob (WebM) or uploaded WAV
        is_recorded = audio_file.filename == 'blob' or audio_file.filename.endswith('.webm')
        
        if is_recorded:
            print("Converting recorded audio...")
            # Add debugging for incoming audio
            audio_segment = AudioSegment.from_file(audio_file)
            print(f"Original audio: channels={audio_segment.channels}, frame_rate={audio_segment.frame_rate}, max_dBFS={audio_segment.max_dBFS}")
            
            # First convert to standard format
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
            
            # Add gain only if max_dBFS is too low
            if audio_segment.max_dBFS < -20:
                audio_segment = audio_segment.apply_gain(10)
            
            # Export with explicit format settings
            audio_segment.export(
                temp_wav,
                format="wav",
                parameters=[
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1"
                ]
            )
            
            # Debug the converted file
            y, sr = librosa.load(temp_wav, sr=16000)
            print(f"Converted audio stats:")
            print(f"Shape: {y.shape}, Sample rate: {sr}")
            print(f"Min/Max before norm: {y.min():.3f}/{y.max():.3f}")
            print(f"RMS value: {np.sqrt(np.mean(y**2)):.3f}")
            
        else:
            print("Processing uploaded WAV...")
            audio_file.save(temp_wav)

        # Process the audio file using your existing function
        predictions, transcription, llm_interpretation = process_audio_file(temp_wav)
        
        return jsonify({
            "Emotion Probabilities": predictions,
            "Transcription": transcription,
            "LLM Interpretation": llm_interpretation
        })

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Cleanup
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)

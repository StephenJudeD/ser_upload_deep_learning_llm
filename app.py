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

def sanitize_audio(audio_segment):
    """Force audio to match SER dataset characteristics"""
    try:
        # Force consistent format
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment = audio_segment.set_sample_width(2)  # 16-bit

        # Convert to numpy array for processing
        audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        
        # Remove DC offset
        audio_array = audio_array - np.mean(audio_array)
        
        # Normalize to match typical SER dataset range
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-10)
        
        # Apply subtle compression to match speech dynamics
        threshold = 0.3
        ratio = 2.0
        audio_array = np.where(
            np.abs(audio_array) > threshold,
            threshold + (np.abs(audio_array) - threshold) / ratio * np.sign(audio_array),
            audio_array
        )
        
        # Convert back to 16-bit PCM
        audio_array = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
        
        # Convert back to AudioSegment
        return AudioSegment(
            audio_array.tobytes(), 
            frame_rate=16000,
            sample_width=2,
            channels=1
        )
    except Exception as e:
        logger.error(f"Error in sanitize_audio: {str(e)}")
        raise

def trim_silences(data, sr, top_db=35):
    """Remove silence from audio"""
    try:
        trimmed_data, _ = librosa.effects.trim(data, top_db=top_db)
        return trimmed_data
    except Exception as e:
        logger.error(f"Error trimming silences: {str(e)}")
        raise

def prepare_audio_length(data, sr, target_duration=3):
    """Prepare audio to be exactly 3 seconds only if it's shorter"""
    target_length = sr * target_duration
    current_length = len(data)
    
    # Only process if audio is shorter than target duration
    if current_length < target_length:
        if len(data) < sr:  # If less than 1 second
            # Use time stretching for very short audio
            data = librosa.effects.time_stretch(data, rate=len(data)/(sr*target_duration))
            # Ensure exact length
            if len(data) > target_length:
                return data[:target_length]
            elif len(data) < target_length:
                return np.pad(data, (0, target_length - len(data)), mode='wrap')
        else:
            # For 1-3 seconds, just pad with repetition
            repetitions = int(np.ceil(target_length / current_length))
            data_repeated = np.tile(data, repetitions)
            return data_repeated[:target_length]
    
    return data  # Return original data if it's 3 seconds or longer

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

def predict_emotion(audio_file, scaler, window_size=3.0, hop_size=0.75):
    """Predict emotions from audio file"""
    try:
        data, sr = librosa.load(audio_file, sr=16000)
        data = trim_silences(data, sr)
        data = normalize_audio(data)
        
        # Add preprocessing for short audio
        data = prepare_audio_length(data, sr, target_duration=3)

        windows = generate_windows(data, window_size, hop_size, sr)

        if len(windows) == 0:
            return {label: "0.00%" for label in label_encoder.classes_}

        emotion_probs = np.zeros(len(label_encoder.classes_))

        for window in windows:
            features = extract_features(window, sr)

            if len(features) != 153:
                raise ValueError(f"Expected feature length of 153, but got {len(features)}")

            features_scaled = scaler.transform(features.reshape(1, -1))
            window_probs = np.mean(
                [predict_with_onnx_model(model, features_scaled) for model in ensemble_models],
                axis=0
            ).squeeze()
            emotion_probs += window_probs

        emotion_probs /= len(windows)

        emotion_probability_distribution = {
            label: f"{prob * 100:.2f}%"
            for label, prob in zip(label_encoder.classes_, emotion_probs)
        }

        return emotion_probability_distribution
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
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
    
                Your task is to provide a very succicnt and insightful interpretation of the emotional content captured in the audio data, considering both the emotion recognition results and the transcript.
    
                In your response, please:
    
                <thinking>
                - Summarize the key emotions detected by the model and their relative strengths.
                - Discuss how the emotions expressed in the transcript align with or differ from the model's predictions.
                - Analyze any notable patterns or trends in the emotional content, especially changes in emotional state over time, differences between speakers, or contextual factors influencing the emotions.
                - Highlight the most salient and informative aspects of the emotional data that would be valuable for understanding the overall emotional experience captured in the audio.
                </thinking>
    
                <result>
                Based on the provided information, your succicnt and insightful interpretation of the emotional content in the audio data is:
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
    audio_file = request.files['audio']
    temp_dir = tempfile.mkdtemp()
    temp_wav = os.path.join(temp_dir, 'audio.wav')
    sanitized_wav = os.path.join(temp_dir, 'sanitized.wav')
    
    try:
        logger.debug(f"Processing file: {audio_file.filename}")
        
        # Check if it's a recorded blob (WebM) or uploaded WAV
        is_recorded = audio_file.filename == 'blob' or audio_file.filename.endswith('.webm')
        
        if is_recorded:
            logger.debug("Converting and sanitizing recorded audio...")
            # Convert to AudioSegment
            audio_segment = AudioSegment.from_file(audio_file)
            
            # Log original audio properties
            logger.debug(f"Original audio: channels={audio_segment.channels}, "
                        f"frame_rate={audio_segment.frame_rate}, "
                        f"max_dBFS={audio_segment.max_dBFS}")
            
            # Sanitize the audio
            audio_segment = sanitize_audio(audio_segment)
            
            # Log sanitized audio properties
            logger.debug(f"Sanitized audio: channels={audio_segment.channels}, "
                        f"frame_rate={audio_segment.frame_rate}, "
                        f"max_dBFS={audio_segment.max_dBFS}")
            
            # Export sanitized audio
            audio_segment.export(
                sanitized_wav,
                format="wav",
                parameters=[
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1"
                ]
            )
            
            # Debug the sanitized file
            y, sr = librosa.load(sanitized_wav, sr=16000)
            logger.debug(f"Sanitized audio stats:")
            logger.debug(f"Shape: {y.shape}, Sample rate: {sr}")
            logger.debug(f"Min/Max: {y.min():.3f}/{y.max():.3f}")
            logger.debug(f"RMS value: {np.sqrt(np.mean(y**2)):.3f}")
            
            process_path = sanitized_wav
            
        else:
            logger.debug("Processing uploaded WAV...")
            audio_file.save(temp_wav)
            # Sanitize uploaded files too for consistency
            audio_segment = AudioSegment.from_wav(temp_wav)
            audio_segment = sanitize_audio(audio_segment)
            audio_segment.export(sanitized_wav, format="wav")
            process_path = sanitized_wav

        # Process the sanitized audio file
        predictions, transcription, llm_interpretation = process_audio_file(process_path)
        
        return jsonify({
            "Emotion Probabilities": predictions,
            "Transcription": transcription,
            "LLM Interpretation": llm_interpretation
        })

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Cleanup
        for file in [temp_wav, sanitized_wav]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)

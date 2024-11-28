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
def assess_audio_quality(audio_data, sr):
    """Assess audio quality based on multiple factors"""
    try:
        # Calculate Signal-to-Noise Ratio (SNR)
        signal_rms = np.sqrt(np.mean(audio_data**2))
        noise_floor = np.sqrt(np.mean(audio_data[audio_data < np.percentile(audio_data, 10)]**2))
        snr = 20 * np.log10(signal_rms / (noise_floor + 1e-10)) if noise_floor > 0 else 0

        # Calculate Dynamic Range
        dynamic_range = 20 * np.log10((np.max(np.abs(audio_data)) + 1e-10) / (np.min(np.abs(audio_data)) + 1e-10))

        # Calculate Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

        # Calculate Peak Amplitude
        peak_amplitude = np.max(np.abs(audio_data))

        # Quality scoring system
        quality_score = 0
        
        # SNR scoring
        if snr > 20:
            quality_score += 2
        elif snr > 15:
            quality_score += 1

        # Dynamic range scoring
        if dynamic_range > 60:
            quality_score += 2
        elif dynamic_range > 40:
            quality_score += 1

        # ZCR scoring
        if 0.05 <= zcr <= 0.15:
            quality_score += 1

        # Peak amplitude scoring
        if 0.5 <= peak_amplitude <= 0.95:
            quality_score += 1

        # Determine quality level
        if quality_score >= 5:
            quality_level = "High"
        elif quality_score >= 3:
            quality_level = "Medium"
        else:
            quality_level = "Low"

        return {
            "level": quality_level,
            "metrics": {
                "snr": round(snr, 2),
                "dynamic_range": round(dynamic_range, 2),
                "zero_crossing_rate": round(float(zcr), 4),
                "peak_amplitude": round(float(peak_amplitude), 4),
                "quality_score": quality_score
            }
        }
    except Exception as e:
        logger.error(f"Error assessing audio quality: {str(e)}")
        return {"level": "Unknown", "metrics": {}}

def check_audio_duration(audio_file):
    """Check audio duration and return appropriate processing strategy"""
    data, sr = librosa.load(audio_file, sr=16000)
    duration = librosa.get_duration(y=data, sr=sr)
    return duration

def get_processing_strategy(duration):
    """Determine processing strategy based on duration"""
    if duration <= 30:  # For files up to 30 seconds
        return "full"
    elif duration <= 300:  # For files up to 5 minutes
        return "chunked"
    else:  # For files longer than 5 minutes
        return "sampled"
        
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
        # Load audio
        data, sr = librosa.load(audio_file, sr=16000)
        
        # Calculate window parameters
        window_length = int(window_size * sr)
        hop_length = int(hop_size * sr)
        
        # Split audio into overlapping windows
        windows = []
        for i in range(0, len(data) - window_length + 1, hop_length):
            window = data[i:i + window_length]
            if len(window) == window_length:
                # 1. Trim silences first
                try:
                    trimmed_window, _ = librosa.effects.trim(window)
                except Exception as trim_error:
                    continue  # Skip this window if trimming fails

                # 2. Normalize after trimming
                normalized_window = normalize_audio(trimmed_window)

                # 3. Ensure 3 seconds duration (pad/truncate)
                if len(normalized_window) > window_length:
                    normalized_window = normalized_window[:window_length]
                else:
                    pad_length = window_length - len(normalized_window)
                    normalized_window = np.pad(normalized_window, (0, pad_length), 'constant')

                windows.append(normalized_window)

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

def get_llm_interpretation(emotional_results, transcription, audio_quality):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"""
            You are an expert in audio emotion recognition and analysis. Given the following information:
    
                Audio data details:
                - Emotional recognition results: {emotional_results}
                - Transcript: {transcription}
                - Audio Quality: {audio_quality['level']} 
                  (SNR: {audio_quality['metrics'].get('snr', 'N/A')}dB, 
                   Dynamic Range: {audio_quality['metrics'].get('dynamic_range', 'N/A')}dB)
    
                Your task is to provide a very succinct and insightful interpretation of the emotional content captured in the audio data, considering the emotion recognition results, transcript, and audio quality.
    
                In your response, please:
    
                <thinking>
                - Start with a brief comment about the audio quality and its potential impact on the analysis
                - Summarize the key emotions detected by the model and their relative strengths
                - Discuss how the emotions expressed in the transcript align with the model's predictions
                - Consider how the audio quality might have affected the emotion recognition accuracy
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
    duration = check_audio_duration(audio_file)
    strategy = get_processing_strategy(duration)
    
    if strategy == "full":
        return process_short_audio(audio_file)
    elif strategy == "chunked":
        return process_chunked_audio(audio_file)
    else:
        return process_sampled_audio(audio_file)

def process_chunked_audio(audio_file):
    """Process longer audio files in chunks"""
    data, sr = librosa.load(audio_file, sr=16000)
    chunk_size = 3 * sr  # 3-second chunks
    overlap = 1 * sr     # 1-second overlap
    
    chunks = []
    predictions = []
    transcriptions = []
    
    for i in range(0, len(data), chunk_size - overlap):
        chunk = data[i:i + chunk_size]
        if len(chunk) >= sr:  # Process only if chunk is at least 1 second
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_chunk:
                sf.write(temp_chunk.name, chunk, sr)
                chunk_pred, chunk_trans, _, _ = process_short_audio(temp_chunk.name)
                predictions.append(chunk_pred)
                transcriptions.append(chunk_trans)
    
    # Aggregate results
    final_prediction = aggregate_predictions(predictions)
    final_transcription = " ".join(transcriptions)
    
    return final_prediction, final_transcription

def aggregate_predictions(predictions):
    """Aggregate predictions from multiple chunks"""
    aggregated = defaultdict(float)
    for pred in predictions:
        for emotion, prob in pred.items():
            aggregated[emotion] += float(prob.strip('%')) / len(predictions)
    
    return {emotion: f"{prob:.2f}%" for emotion, prob in aggregated.items()}

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    temp_dir = tempfile.mkdtemp()
    temp_wav = os.path.join(temp_dir, 'audio.wav')
    
    try:
        print(f"Processing file: {audio_file.filename}")
        
        is_recorded = audio_file.filename == 'blob' or audio_file.filename.endswith('.webm')
        
        if is_recorded:
            print("Converting recorded audio...")
            audio_segment = AudioSegment.from_file(audio_file)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
            
            if audio_segment.max_dBFS < -35:
                audio_segment = audio_segment.normalize()
            
            audio_segment.export(
                temp_wav,
                format="wav",
                parameters=[
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1"
                ]
            )
            
        else:
            print("Processing uploaded WAV...")
            audio_file.save(temp_wav)

        predictions, transcription, llm_interpretation, audio_quality = process_audio_file(temp_wav)
        
        return jsonify({
            "Emotion Probabilities": predictions,
            "Transcription": transcription,
            "LLM Interpretation": llm_interpretation,
            "Audio Quality": audio_quality
        })

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)

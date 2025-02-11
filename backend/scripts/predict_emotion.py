import librosa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import json
import os
import sys
import pickle
import warnings
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Suppress version warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set TensorFlow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def extract_features(file_path, max_len=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        duration = len(audio) / sample_rate
        # Removed print statement to avoid non-JSON output

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccs = np.mean(mfccs.T, axis=0)

        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma = np.mean(chroma.T, axis=0)

        # Extract Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast = np.mean(contrast.T, axis=0)

        # Extract Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr = np.mean(zcr.T, axis=0)

        # Extract Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        centroid = np.mean(centroid.T, axis=0)

        # Extract Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85)
        rolloff = np.mean(rolloff.T, axis=0)

        # Extract RMS Energy
        rms = librosa.feature.rms(y=audio)
        rms = np.mean(rms.T, axis=0)

        # Concatenate features
        features = np.concatenate([mfccs, chroma, contrast, zcr, centroid, rolloff, rms])

        # Ensure fixed length
        if len(features) < max_len:
            features = np.pad(features, (0, max_len - len(features)), mode='constant')
        elif len(features) > max_len:
            features = features[:max_len]

        return features, audio, sample_rate
    except Exception as e:
        error_msg = {"error": f"Error processing {file_path}: {str(e)}"}
        print(json.dumps(error_msg))
        return None, None, None

def generate_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def save_mel_spectrogram(audio, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    img_str = generate_base64_image(fig)
    plt.close(fig)
    return img_str

def save_polar_plot(emotion_probabilities):
    emotions = list(emotion_probabilities.keys())
    probabilities = list(emotion_probabilities.values())

    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    angles += angles[:1]
    probabilities += probabilities[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.fill(angles, probabilities, color='skyblue', alpha=0.4)
    ax.plot(angles, probabilities, color='blue', linewidth=2)

    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="gray", fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions, fontsize=12, color="darkblue")

    plt.title("Emotion Probabilities", va='bottom', fontsize=14, color="darkblue")

    img_str = generate_base64_image(fig)
    plt.close(fig)
    return img_str

def predict_emotion(audio_path, model, label_encoder, max_len=40):
    """
    Predict emotion from an audio file with improved error handling and feature extraction.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Extract features from the audio file
        features, audio, sample_rate = extract_features(audio_path, max_len=max_len)
        if features is None:
            raise RuntimeError("Feature extraction failed")

        # Reshape for model input
        features = np.expand_dims(features, axis=0)

        # Make prediction
        predictions = model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]

        # Calculate probabilities
        emotion_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob * 100)
            for i, prob in enumerate(predictions[0])
        }

        return predicted_emotion, emotion_probabilities, audio, sample_rate

    except Exception as e:
        error_msg = {"error": f"Prediction failed: {str(e)}"}
        print(json.dumps(error_msg))
        return None, None, None, None

def main():
    """
    Entry point for the script. Reads the audio file path from command-line arguments and performs prediction.
    """
    try:
        # Get audio path from command-line argument
        if len(sys.argv) < 2:
            raise ValueError("Audio file path not provided.")

        audio_path = sys.argv[1]

        # Define paths for model and label encoder
        base_dir = r"C:\Users\91938\OneDrive\Documents\Desktop\emotion-recognizer"
        model_path = os.path.join(base_dir, "backend", "models", "ann_new_emotion_recognition_model.h5")
        label_encoder_path = os.path.join(base_dir, "backend", "models", "new_label_encoder (1).pkl")

        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)

        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Make prediction
        emotion, probabilities, audio, sample_rate = predict_emotion(audio_path, model, label_encoder)

        if emotion is None:
            return

        # Generate mel spectrogram
        mel_spectrogram_base64 = save_mel_spectrogram(audio, sample_rate)

        # Generate polar plot
        polar_plot_base64 = save_polar_plot(probabilities)

        # Output result
        result = {
            "emotion": emotion,
            "probabilities": probabilities,
            "melSpectrogramBase64": mel_spectrogram_base64,
            "polarPlotBase64": polar_plot_base64
        }
        print(json.dumps(result, indent=4))

    except Exception as e:
        # Output error message
        error_result = {"error": str(e)}
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()

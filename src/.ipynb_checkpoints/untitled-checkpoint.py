import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from transformers import pipeline
import whisper
import librosa
import soundfile as sf
import io
from audio_recorder_streamlit import audio_recorder  # Changed import here

# Define a function to extract MFCC features from the audio file
def extract_mfcc(audio, sr, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed
    
# Load your trained emotion recognition model
@st.cache_resource
def load_emotion():
    return load_model('/Users/fahad/Desktop/T5/src/test1.h5')

model = load_emotion()

# Define the mapping from predicted index to emotion label
emotion_labels = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
    # Add more if you have more classes
}

# Streamlit UI setup
st.title('Audio Emotion Recognition and Text Detection')

# Initialize Whisper model for transcription
@st.cache_resource
def load_whiaper():
    return whisper.load_model("small")
text_detection_model = load_whiaper()  


# Initialize department classifier
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

department_classifier =load_zero_shot()
department_labels = ['Civil', 'Police', 'Traffic', 'Ambulance']

# File uploader for pre-recorded audio files
st.subheader("Upload an audio file")
uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3'])

# Audio Recorder component for live audio recording
st.subheader("Or record your voice")
audio_bytes = audio_recorder()  # Changed function here

# Placeholder for messages
message = st.empty()
def process_audio(audio_data, file_extension):
    try:
        # Convert the audio bytes to a NumPy array
        if file_extension == 'wav':
            # If it's a WAV file, we can read it directly with soundfile
            audio, sample_rate = sf.read(io.BytesIO(audio_data))
        else:
            # If it's not a WAV file, use librosa to load it which will also convert it
            audio, sample_rate = librosa.load(io.BytesIO(audio_data), sr=None)

        # Ensure audio is a NumPy array
        #assert isinstance(audio, np.ndarray), "audio is not an np.ndarray"

        # Extract features from the audio array
        features = extract_mfcc(audio, sample_rate)  # Pass audio data and sample rate
        features = features.reshape(1, -1)

        # Make emotion prediction
        emotion_prediction = model.predict(features)
        predicted_emotion_index = np.argmax(emotion_prediction)
        predicted_emotion = emotion_labels[predicted_emotion_index]

        # Display predicted emotion
        st.write(f'Predicted Emotion: {predicted_emotion}')

        # Transcribe the audio using Whisper
        transcribed_text = text_detection_model.transcribe(io.BytesIO(audio_data))["text"]

        # Display transcribed text
        st.write(f'Transcribed Text: {transcribed_text}')

        # Classify the department based on the transcribed text
        department_result = department_classifier(transcribed_text, department_labels)
        predicted_department = department_result['labels'][0]

        # Display predicted department
        st.write(f'Predicted Department: {predicted_department}')

    except Exception as e:
        message.error(f'Error processing audio file: {e}')

    except Exception as e:
        message.error(f'Error processing audio file: {e}')
# Check if a file has been uploaded and process it
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    st.audio(uploaded_file, format='audio/wav')
    process_audio(uploaded_file.read(), file_extension)

# Check if an audio has been recorded and process it
if audio_bytes is not None:
    st.audio(audio_bytes, format='audio/wav')
    process_audio(audio_bytes, 'wav')  # Assuming the recorded audio is in 'wav' format
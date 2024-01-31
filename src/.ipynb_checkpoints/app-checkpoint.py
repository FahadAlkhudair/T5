import streamlit as st
import librosa
import numpy as np
from keras.models import load_model
st.markdown(
    """
    <style>
        .title {
            color: #00BFFF;
            font-size: 36px;
            font-weight: bold;
        }

        .subheader {
            color: #00BFFF;
            font-size: 24px;
            font-weight: bold;
        }

        .waveform-container {
            display: flex;
            flex-direction: row;1§
        }

        .waveform {
            flex-grow: 1;
        }

        .sample-rate {
            margin-top: 20px;
            color: #00BFFF;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# Hello, Bootcampers! :flag-sa:")
st.markdown("## Audio recognition Capstone")
st.markdown("### T5 has :blue[Finished] 😍")
st.markdown("## Let's get started", unsafe_allow_html=True)


st.markdown("## Try it yourself")

# Load your trained model (make sure the path is correct)
model = load_model('/Users/irk2w/Desktop/T5/src/test1.h5')

# Define the mapping from predicted index to emotion label
# This should match the order of your model's output
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

# Function to extract MFCC features
def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

st.title('Emotion Recognition from Audio')

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Save buffer to a temporary file
    with open('temp_audio.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        # Extract features and reshape it for the model
        features = extract_mfcc('temp_audio.wav')
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_emotion = emotion_labels[predicted_index[0]]
        
        # Display the prediction
        st.write(f'Predicted Emotion: {predicted_emotion}')
        
    except Exception as e:
        st.error(f'Error processing audio file: {e}')
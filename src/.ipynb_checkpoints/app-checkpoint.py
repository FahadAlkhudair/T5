import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

model = load_model('/Users/irk2w/Desktop/T5/src/test1.h5')


emotion_labels = {
   1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fearful',7:'disgust',8:'surprised'}

def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

st.title('Emotion Recognition from Audio')

uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open('temp_audio.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        features = extract_mfcc('temp_audio.wav')
        features = features.reshape(1, -1)
        
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_emotion = emotion_labels[predicted_index[0]]
        
        st.write(f'Predicted Emotion: {predicted_emotion}')
        
    except Exception as e:
        st.error(f'Error processing audio file: {e}')
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
import io
import tensorflow as tf
import librosa
import numpy as np

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
            flex-direction: row;
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

st.markdown("# Hello, *Bootcampers!* :flag-sa:")
st.markdown("## Audio recognition Capstone")
st.markdown("### _T5_ has :blue[Finished] :heart_eyes:")
st.markdown("## Let's get started", unsafe_allow_html=True)


st.markdown("## Try it yourself")

audio_file = st.file_uploader("Upload recorded audio", type=[".wav", "mp3"])

if audio_file is not None:
    audio_data, samplerate = sf.read(audio_file)

    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio_data, samplerate, format="wav")

    st.audio(audio_bytes, format="audio/wav")

    st.markdown("### Audio Waveform")
    st.write(audio_data)

    st.markdown("### Sample Rate")
    st.write(samplerate)

    model = tf.keras.models.load_model("test1.h5")

    # Convert audio data to mono if it has more than one channel
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Resample audio if necessary
    if samplerate != 16000:
        audio_data = librosa.core.resample(audio_data, orig_sr=samplerate, target_sr=16000)

    # Normalize audio data
    audio_data = audio_data / np.abs(audio_data).max()

    # Convert audio data to spectrogram
    n_fft = 2048  # Number of FFT points for spectrogram calculation
    hop_length = 512  # Number of samples between successive frames
    n_mels = 128  # Number of mel frequency bands
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Resize the spectrogram to match the desired shape
    desired_shape = (40, 1)
    if spectrogram.shape[1] < desired_shape[0]:
        # Pad the spectrogram if it is shorter than the desired shape
        pad_width = desired_shape[0] - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)))
    elif spectrogram.shape[1] > desired_shape[0]:
        # Crop the spectrogram if it is longer than the desired shape
        spectrogram = spectrogram[:, :desired_shape[0]]

    # Add an extra dimension for the channel
    resized_spectrogram = np.expand_dims(spectrogram, axis=-1)

    # Make predictions with the resized spectrogram
    predictions = model.predict(resized_spectrogram)
    print("Predictions:", predictions)

    # Get the predicted index
    predicted_index = np.argmax(predictions)
    print("Predicted Index:", predicted_index)

    # Define the emotions mapping
    emotions = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }

    # Get the predicted emotion from the dictionary
    predicted_emotion = emotions.get(predicted_index)
    print("Predicted Emotion:", predicted_emotion)

    st.markdown("### Predictions")
    st.write("Predicted Emotion:", predicted_emotion)

# Link the CSS file
st.markdown(
    """
    <style>
        @import url('./style.css');
    </style>
    """,
    unsafe_allow_html=True
)
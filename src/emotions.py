import os
import datetime
import whisper
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
from langdetect import detect

# Load the Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("large-v3")

# Load the zero-shot classification model
@st.cache_resource
def load_classifier():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

# Load the emotion classification model
@st.cache_resource
def load_emotion_classifier():
    return pipeline("audio-classification", model="harshit345/xlsr-wav2vec-speech-emotion-recognition")

text_detection_model = load_whisper()
zero_shot_classifier = load_classifier()
emotion_classifier = load_emotion_classifier()

# Function to save audio bytes to a file
def save_audio_file(audio_bytes, file_extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name

# Function to transcribe audio file and return detected language
def transcribe(audio_file_path):
    result = text_detection_model.transcribe(audio_file_path)
    return result

# Function to transcribe audio and get detected language
def transcribe_audio(file_path):
    transcript = transcribe(file_path)
    text = transcript["text"]
    language = detect(text)
    return text, language

# Function to classify the transcript
def classify_transcript(text):
    classes = ["الدفاع المدني", "المرور", "الاسعاف", "الشرطة"]
    result = zero_shot_classifier(text, candidate_labels=classes, hypothesis_template="This is a {} call.")
    return result

# Function to classify emotion of audio
def classify_emotion(audio_file_path):
    with open(audio_file_path, "rb") as f:
        audio_data = f.read()
    result = emotion_classifier(audio_data)
    return result

def main():
    st.title("بلاغك")

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    # Record Audio tab
    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            audio_file = save_audio_file(audio_bytes, "wav")
            st.audio(audio_bytes, format="audio/wav")

    # Upload Audio tab
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])
        if audio_file:
            file_extension = audio_file.type.split('/')[1]
            audio_file_path = save_audio_file(audio_file.read(), file_extension)

    # Transcribe, classify, and detect emotion button action
    if st.button("Transcribe and Classify"):
        # Find the newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )

        # Transcribe the audio file
        transcript_text, detected_language = transcribe_audio(audio_file_path)

        # Display the transcript and detected language
        st.header("Transcript")
        st.write(transcript_text)
        
        st.header("Detected Language")
        st.write(detected_language)

        # Classify the transcript
        classification_result = classify_transcript(transcript_text)
        st.header("Classification")
        st.write(f"Predicted Department: {classification_result['labels'][0]}")

        # Classify the emotion
        emotion_result = classify_emotion(audio_file_path)
        st.header("Emotion Classification")
        st.write(f"Detected Emotion: {emotion_result[0]['label']} - Confidence: {emotion_result[0]['score']*100:.2f}%")

        # Save the transcript to a text file
        with open(f"{audio_file_path.split('.')[0]}.txt", "w") as f:
            f.write(transcript_text)

        # Provide a download button for the transcript
        st.download_button(
            label="Download Transcript",
            data=transcript_text,
            file_name=f"{audio_file_path.split('.')[0]}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
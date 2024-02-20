import os
import datetime
import whisper
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Function to load the Whisper model
@st.cache(allow_output_mutation=True)
def load_whisper():
    return whisper.load_model("large-v3")

# Function to load the zero-shot classification model
@st.cache
def load_classifier():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

text_detection_model = load_whisper()
zero_shot_classifier = load_classifier()

# Function to save audio bytes to a file
def save_audio_file(audio_bytes, file_extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name

# Function to transcribe audio file
def transcribe(audio_file_path):
    result = text_detection_model.transcribe(audio_file_path)
    return result

# Function to transcribe audio
def transcribe_audio(file_path):
    transcript = transcribe(file_path)
    return transcript["text"]

# Function to classify the transcript
def classify_transcript(text):
    classes = ["الدفاع المدني", "المرور", "الاسعاف", "الشرطة"]
    result = zero_shot_classifier(text, candidate_labels=classes, hypothesis_template="This is a {} call.")
    return result

def main():
    st.set_page_config(page_title="بلاغك", page_icon="Desktop/T5/src/logo.png")

    # Add logo and title
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image("Desktop/T5/src/logo.png", use_column_width=True)
    with col2:
        st.title("بلاغك")

    # Add image in the top corner
    st.image("Desktop/T5/src/KSA.jpeg", use_column_width=False)

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

    # Transcribe and classify button action
    if st.button("Transcribe and Classify"):
        # Find the newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )

        # Transcribe the audio file
        transcript_text = transcribe_audio(audio_file_path)

        # Display the transcript
        st.header("Transcript")
        st.write(transcript_text)

        # Classify the transcript
        classification_result = classify_transcript(transcript_text)
        st.header("Classification")
        st.write(f"Predicted Department: {classification_result['labels'][0]}")

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
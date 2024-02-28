import os
import datetime
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
from langdetect import detect
import whisper
# Load the Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("small")

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
    rtl_and_custom_font_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Arabic:wght@400;700&display=swap');

        body {
            direction: rtl;
            text-align: right;
            font-family: "Arial", sans-serif;
        }
        .stApp {
            background-image: url('https://c.top4top.io/p_2977qjjb71.png'); /* Add your image URL */
            background-attachment: fixed;
            background-position: center;
            direction: rtl;
            text-align: right;
            background-size: 100vw 100vh;  /* This sets the size to cover 100% of the viewport width and height */
        }
        .title-text {
            color: white;
            font-family: 'IBM Plex Arabic', sans-serif;
            margin-right: 10px;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1rem;
        }
    </style>
    """
    st.markdown(rtl_and_custom_font_style, unsafe_allow_html=True)



    tab1, tab2 = st.tabs(["سجل صوتية", "ارفع صوتية"])

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
    if st.button("ابدأ"):
        # Find the newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )

        # Transcribe the audio file
        transcript_text, detected_language = transcribe_audio(audio_file_path)

        # Display the transcript and detected language
        #st.header("Transcript")
        #st.write(transcript_text)
        
        st.header("اللغة")
        st.write(detected_language)

        # Classify the transcript
        classification_result = classify_transcript(transcript_text)
        st.header("تصنيف البلاغ")
        st.write(f"القطاع: {classification_result['labels'][0]}")

        # Classify the emotion
        emotion_result = classify_emotion(audio_file_path)
        st.header("شعور المتصل")
        st.write(f"الشعور: {emotion_result[0]['label']}")

        # Save the transcript to a text file
        with open(f"{audio_file_path.split('.')[0]}.txt", "w") as f:
            f.write(transcript_text)

        # Provide a download button for the transcript
        st.download_button(
            label="تحميل الكتابة",
            data=transcript_text,
            file_name=f"{audio_file_path.split('.')[0]}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
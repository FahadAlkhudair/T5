## T5 Capstone Project Overview: Blaghk (بلاغك) 🎧

Welcome to the GitHub repository for our T5 Capstone Project, *"Blaghk" (بلاغك)*, an Artificial Intelligence-driven tool designed to streamline the process of directing reports to the appropriate emergency services. Our goal is to ensure a prompt and accurate response, leveraging the power of AI in audio classification.

## Team Members 👥

- Fahad Alkhudair 🧑‍💼 (https://github.com/FahadAlkhudair)
- Rakan Alkhuraiji 🧑‍💼 (https://github.com/Rk2w)
- Naif Alhmoud 🧑‍💼 (https://github.com/Naif-Alhamoud)
- Mohammed Alzubaidi 🧑‍💼 (https://github.com/fanciful-95)
- Mohammed Alsubaie 🧑‍💼 (https://github.com/mohammmedsub)

## Project Description 📖

In emergency situations, every second counts. Blaghk (بلاغك) is an AI tool that classifies audio inputs to identify the nature of the emergency. It then directs the report to the corresponding emergency service, such as medical aid, fire department, or police assistance. This intelligent system minimizes the time taken to relay critical information, thereby expediting the response time of emergency services.

The dataset used for training our AI model has been meticulously recorded and transcribed by our dedicated team members, ensuring a robust and diverse collection of audio samples for accurate classification.

## Technology Stack 💻

We have utilized a combination of technologies to bring Blaghk (بلاغك) to life:

- **Streamlit**: An open-source app framework that is the cornerstone of our user interface, providing a seamless and interactive front-end experience. 🖥️
- **Jupyter Notebook**: A web-based interactive computing platform used for the development of our Python-based backend. It has been instrumental in prototyping and testing our AI models. 📓

## Methodology 💯

The application follows a clear workflow to provide a seamless user experience:

1. **Recording or Uploading Audio**: Users begin by either recording their voice directly in the app or uploading an audio file in a supported format (e.g., mp3, wav).
2. **Saving and Processing Audio**: The audio is then saved with a timestamp to ensure uniqueness and processed using the Whisper model for transcription.
3. **Language Detection**: The resulting transcription is used for language detection via the `langdetect` library, identifying the language of the spoken content.
4. **Classification**: The transcribed text is classified into predefined categories using a zero-shot classification model to understand the context or intent of the speech.
5. **Emotion Recognition**: Meanwhile, the original audio is analyzed by an emotion classification model to capture the speaker's emotional state.
6. **Results Presentation**: The application displays the transcription, detected language, classification, and emotion recognition results to the user.
7. **Transcript Download**: Users can also download the transcribed text as a `.txt` file for their reference.

Each of these steps involves machine learning models that have been carefully chosen and integrated into the app to provide accurate and useful results.

## How to Use Blaghk (بلاغك) 🔍
Demo of the website (https://drive.google.com/file/d/1tgCsNOChLXzPBFPLBhHMTRkqlgZMX906/view?usp=sharing)


---

We are committed to continual improvement and eagerly anticipate feedback from the users of Blaghk (بلاغك). Our team is passionate about harnessing the capabilities of AI to serve communities and save lives.

*Thank you for your interest in our project, and we hope it serves as a valuable tool for emergency response teams in Saudi Arabia.* 🇸🇦 

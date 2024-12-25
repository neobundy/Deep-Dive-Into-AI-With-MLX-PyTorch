# Standard library imports
import os
import whisper
import streamlit as st
import requests



TEMP_FOLDER = "./temp"
IMAGE_FOLDER = "./images"
MODEL_FOLDER = "./models"
# DEFAULT_MODEL = MODEL_FOLDER + "/medium"
# DEFAULT_MODEL = "mlx-community/whisper-medium"

TRANSCRIPTION_TEMP_AUDIO_FILE = TEMP_FOLDER + "/transcription_temp_audio.wav"
AUDIO_SERVER_URL = "http://localhost:5000"
AUDIO_SERVER_PORT = 5000
PROJECT_TITLE = "Whispering MLX Chatbot Ideation"
AVATAR_AI = IMAGE_FOLDER + "/ai.png"


def init_page():
    st.set_page_config(page_title=PROJECT_TITLE, page_icon="🤗")
    st.header(f"{PROJECT_TITLE}", anchor="top")

    # st.image(AVATAR_AI)

    st.markdown(""" [Go to Bottom](#bottom) """, unsafe_allow_html=True)
    st.sidebar.subheader("Options")


def init_session_state():
    st.session_state.setdefault("transcript", "")


def display_tts_panel():
    tts_options = st.sidebar.expander("TTS/STT")
    with tts_options:
        st.session_state.use_audio = st.checkbox("Use Audio", value=False)
        if st.button("Record"):
            r = requests.get(f"{AUDIO_SERVER_URL}/start_recording")
            if r.status_code == 200:
                st.success("Start recording...")
            else:
                st.error("Couldn't start recording.")
        if st.button("Stop"):
            r = requests.get(f"{AUDIO_SERVER_URL}/stop_recording")
            if r.status_code == 200:
                st.success("Successfully recorded.")
                # st.session_state.transcript = whisper.transcribe(TRANSCRIPTION_TEMP_AUDIO_FILE, path_or_hf_repo=DEFAULT_MODEL)["text"]
                st.session_state.transcript = whisper.transcribe(TRANSCRIPTION_TEMP_AUDIO_FILE)["text"]
            else:
                st.error("Couldn't stop recording.")


def setup_and_cleanup(func):
    # TODO: pre-processing
    print("Do your pre-processing here")

    # Ensure the directories exist
    os.makedirs(os.path.dirname(TEMP_FOLDER), exist_ok=True)
    os.makedirs(os.path.dirname(IMAGE_FOLDER), exist_ok=True)

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    # TODO: post-processing

    print("Do your post-processing here")

    return wrapper


def display_chatbot_panel():
    if st.session_state.transcript:
        st.markdown(st.session_state.transcript)

# pre- and post- processor decorator for main function
@setup_and_cleanup
def main():
    init_page()
    init_session_state()
    display_tts_panel()
    display_chatbot_panel()


if __name__ == "__main__":
    main()

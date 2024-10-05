import os
import subprocess
import tempfile
import uuid
import streamlit as st
import base64
import time

# Constants
BASE_DIR = os.path.dirname(__file__)
OUTPUT_WAV_PATH = os.path.join(BASE_DIR, 'output_wav')
TRANSCRIPTION_PATH = os.path.join(BASE_DIR, 'transcription')
VOICE_DIR = os.path.join(BASE_DIR, 'voices')
TOAST_DURATION = 10

# Setup functions
def setup_directories():
    os.makedirs(OUTPUT_WAV_PATH, exist_ok=True)
    os.makedirs(TRANSCRIPTION_PATH, exist_ok=True)

# Utility functions
def toast(msg, duration):
    st.toast(msg)
    time.sleep(duration)

# Audio processing functions
def clean_audio(speaker_wav, voice_cleanup):
    if voice_cleanup:
        try:
            cleaned_audio = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + ".wav")
            lowpass_highpass = "lowpass=8000,highpass=75,"
            trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
            shell_command = ["ffmpeg", "-y", "-i", speaker_wav, "-af", f"{lowpass_highpass}{trim_silence}", cleaned_audio]
            process = subprocess.Popen(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                toast(f"Error during filtering: {stderr.decode()}", duration=3)
                return speaker_wav
            return cleaned_audio
        except Exception as e:
            toast(f"Error filtering audio: {str(e)}", TOAST_DURATION)
            return speaker_wav
    return speaker_wav

def process_audio(input_path, output_path, audio_quality, voice_speed):
    speed_values = {"Slow": "0.75", "Normal": "1.0", "Fast": "1.25"}
    tempo = speed_values[voice_speed]

    quality_settings = {
        "Low": (16000, "16"),
        "Medium": (24000, "24"),
        "High": (44100, "32")
    }
    sample_rate, bit_depth = quality_settings[audio_quality]

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", f"atempo={tempo}",
        "-ar", str(sample_rate),
        "-acodec", f"pcm_s{bit_depth}le",
        output_path
    ])

def save_transcription(prompt, output_path):
    text_filename = os.path.splitext(os.path.basename(output_path))[0] + ".txt"
    text_file_path = os.path.join(TRANSCRIPTION_PATH, text_filename)
    with open(text_file_path, "w") as text_file:
        text_file.write(prompt)

# UI functions
def show_previous_files():
    st.subheader("Generated Files")
    audio_files = [f for f in os.listdir(OUTPUT_WAV_PATH) if f.endswith('.wav')]
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_WAV_PATH, f)), reverse=True)
    
    for audio_file in audio_files:
        audio_file_path = os.path.join(OUTPUT_WAV_PATH, audio_file)
        text_file_path = os.path.join(TRANSCRIPTION_PATH, os.path.splitext(audio_file)[0] + ".txt")

        if os.path.exists(text_file_path):
            with open(text_file_path, "r") as text_file:
                prompt_text = text_file.read()

            with st.expander(audio_file, expanded=False):
                st.audio(audio_file_path, format="audio/wav")
                st.write("Prompt used:", prompt_text)
                st.write("Date generated:", time.ctime(os.path.getmtime(audio_file_path)))

                if st.button("Delete", key=audio_file):
                    try:
                        os.remove(audio_file_path)
                        os.remove(text_file_path)
                        toast(f"{audio_file} has been deleted.", TOAST_DURATION)
                    except Exception as e:
                        toast(f"Error deleting {audio_file}: {str(e)}", TOAST_DURATION)

def main():
    st.set_page_config(page_title="EpisodeX", page_icon="x.png", layout="centered")
    st.title("EpisodeX")
    st.markdown("Process and manage audio files.")

    setup_directories()

    input_option = st.radio("Choose input option", ("Type Text", "Upload Text File"))
    prompt, char_count = get_input_text(input_option)
    st.markdown(f"**Character count: {char_count}/2000**")
    
    speaker_wav, voice_cleanup = get_voice_input()
    voice_speed = st.selectbox("Select Voice Speed", options=["Slow", "Normal", "Fast"])
    audio_quality = st.selectbox("Select Audio Quality", options=["Low", "Medium", "High"])

    if st.button("Process Audio"):
        if not speaker_wav:
            toast("Please upload an audio file.", TOAST_DURATION)
            return
        
        cleaned_audio = clean_audio(speaker_wav, voice_cleanup)
        output_path = os.path.join(OUTPUT_WAV_PATH, f"processed_{os.path.basename(speaker_wav)}")
        process_audio(cleaned_audio, output_path, audio_quality, voice_speed)
        save_transcription(prompt, output_path)
        display_audio_output(output_path)

    show_previous_files()

def get_input_text(input_option):
    if input_option == "Type Text":
        prompt = st.text_area("Text Prompt", placeholder="Enter your text here", height=200)
    elif input_option == "Upload Text File":
        text_file = st.file_uploader("Upload a text file", type=["txt"])
        prompt = text_file.read().decode("utf-8") if text_file else ""
    return prompt, len(prompt)

def get_voice_input():
    voice_path = [f for f in os.listdir(VOICE_DIR) if f.endswith('.wav')]
    voice_option = st.radio("Choose voice input option", ("Select Voice", "Upload Voice File"))

    if voice_option == "Select Voice":
        selected_voice = st.selectbox("Choose a pre-loaded voice", voice_path)
        speaker_wav = os.path.join(VOICE_DIR, selected_voice)
        voice_cleanup = False
    elif voice_option == "Upload Voice File":
        voice_file = st.file_uploader("Upload voice file", type=['wav'])
        voice_cleanup = st.checkbox("Cleanup Reference Voice", help="Apply noise filtering to reference audio.")
        if voice_file:
            voice_data = voice_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(voice_data)
                speaker_wav = temp_file.name
        else:
            speaker_wav = None

    return speaker_wav, voice_cleanup

def display_audio_output(audio_output):
    st.audio(audio_output, format="audio/wav")
    
    st.download_button(
        label="Download Audio",
        data=open(audio_output, "rb").read(),
        file_name=os.path.basename(audio_output),
        mime="audio/wav"
    )
    
    toast(f"Audio processed and saved to {audio_output}", TOAST_DURATION)

if __name__ == "__main__":
    main()

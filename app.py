import os
import subprocess
import tempfile
import uuid
import torch
import streamlit as st
import torchaudio
import base64
import io
import time
from TTS.api import TTS

# Constants
BASE_DIR = os.path.dirname(__file__)
OUTPUT_WAV_PATH = os.path.join(BASE_DIR, 'output_wav')
TRANSCRIPTION_PATH = os.path.join(BASE_DIR, 'transcription')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xtts_v2')
FFMPEG_PATH = os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffmpeg.exe')
VOICE_DIR = os.path.join(BASE_DIR, 'voices')
TOAST_DURATION = 10

# Set ffmpeg path for pydub
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

from pydub import AudioSegment

# Language options
LANGUAGE_OPTIONS = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Russian": "ru"
}

# Setup functions
def setup_directories():
    os.makedirs(OUTPUT_WAV_PATH, exist_ok=True)
    os.makedirs(TRANSCRIPTION_PATH, exist_ok=True)

def load_tts_model():
    if 'tts_model' not in st.session_state:
        try:
            config_path = os.path.join(MODEL_PATH, 'config.json')
            model_file = os.path.join(MODEL_PATH, 'model.pth')
            
            for path in [config_path, model_file]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} not found.")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.session_state.tts_model = TTS(model_path=MODEL_PATH, config_path=config_path).to(device)
            toast("TTS model loaded successfully!", TOAST_DURATION)
        except Exception as e:
            toast(f"Error loading TTS model: {e}", TOAST_DURATION)
            st.stop()

# Utility functions
def toast(msg, duration):
    st.toast(msg)
    time.sleep(duration)

def plot_wf(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.squeeze().numpy()
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())
    
    width, height, padding, line_width = 800, 200, 10, 1
    resampled = [waveform[int(i * len(waveform) / width)] for i in range(width)]
    
    path_data = "M" + " ".join([f"{i + padding},{int((1 - sample) * (height - 2 * padding) + padding)}" for i, sample in enumerate(resampled)])
    
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><path d="{path_data}" fill="none" stroke="SteelBlue" stroke-width="{line_width}"/></svg>'
    svg_bytes = svg.encode('utf-8')
    base64_svg = base64.b64encode(svg_bytes).decode('utf-8')
    
    return f"data:image/svg+xml;base64,{base64_svg}"

# Audio processing functions
def clean_audio(speaker_wav, voice_cleanup):
    if voice_cleanup:
        try:
            cleaned_audio = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + ".wav")
            lowpass_highpass = "lowpass=8000,highpass=75,"
            trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
            shell_command = [FFMPEG_PATH, "-y", "-i", speaker_wav, "-af", f"{lowpass_highpass}{trim_silence}", cleaned_audio]
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

def generate_speech(prompt, language, speaker_wav, voice_cleanup, audio_quality, voice_speed):
    speaker_wav = clean_audio(speaker_wav, voice_cleanup)
    
    try:
        wav_output = st.session_state.tts_model.tts(prompt, speaker_wav=speaker_wav, language=language)
        wav_tensor = torch.tensor(wav_output).unsqueeze(0)

        base_output_path = os.path.join(OUTPUT_WAV_PATH, "wav")
        temp_output_path = f"{base_output_path}_temp.wav"
        output_path = get_unique_output_path(base_output_path)

        torchaudio.save(temp_output_path, wav_tensor, 24000)

        apply_audio_settings(temp_output_path, output_path, audio_quality, voice_speed)

        os.remove(temp_output_path)

        save_transcription(prompt, output_path)

        return output_path
    except Exception as e:
        toast(f"Error generating speech: {str(e)}", TOAST_DURATION)
        return None

def get_unique_output_path(base_output_path):
    output_path = f"{base_output_path}.wav"
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base_output_path}{counter}.wav"
        counter += 1
    return output_path

def apply_audio_settings(input_path, output_path, audio_quality, voice_speed):
    speed_values = {"Slow": "0.75", "Normal": "1.0", "Fast": "1.25"}
    tempo = speed_values[voice_speed]

    quality_settings = {
        "Low": (16000, "16"),
        "Medium": (24000, "24"),
        "High": (44100, "32")
    }
    sample_rate, bit_depth = quality_settings[audio_quality]

    subprocess.run([
        FFMPEG_PATH, "-y",
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
    st.markdown("Generate speech from text using a powerful TTS model.")

    setup_directories()
    load_tts_model()

    input_option = st.radio("Choose input option", ("Type Text", "Upload Text File"))
    prompt, char_count = get_input_text(input_option)
    st.markdown(f"**Character count: {char_count}/2000**")
    
    language = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), index=0)
    speaker_wav, voice_cleanup = get_voice_input()
    voice_speed = st.selectbox("Select Voice Speed", options=["Slow", "Normal", "Fast"])
    audio_quality = st.selectbox("Select Audio Quality", options=["Low", "Medium", "High"])

    if st.button("Generate Speech"):
        if not prompt:
            toast("Please enter some text or upload a text file.", TOAST_DURATION)
            return
        
        if not speaker_wav:
            toast("Please upload a reference audio file.", TOAST_DURATION)
            return
        
        audio_output = generate_speech(prompt, LANGUAGE_OPTIONS[language], speaker_wav, voice_cleanup, audio_quality, voice_speed)

        if audio_output:
            display_audio_output(audio_output)

    if 'audio_output' not in locals():
        display_temporary_waveform()

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
        voice_file = st.file_uploader("Upload voice file (for voice cloning)", type=['wav'])
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
    
    st.markdown("### Waveform")
    actual_waveform_img = plot_wf(audio_output)
    st.markdown(f'<img src="{actual_waveform_img}" alt="Waveform" style="max-width: 100%;">', unsafe_allow_html=True)

    toast(f"Speech generated and saved to {audio_output}", TOAST_DURATION)

def display_temporary_waveform():
    st.markdown("### Waveform")
    temporary_waveform = '<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg"><rect width="800" height="200" style="fill:none;" /></svg>'
    st.markdown(f'<img src="data:image/svg+xml;base64,{base64.b64encode(temporary_waveform.encode()).decode("utf-8")}" alt="Temporary Waveform" style="max-width: 100%;">', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

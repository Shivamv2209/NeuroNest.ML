import os
import sys
import google.generativeai as genai
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import tempfile
import queue
import threading
import time
import pyttsx3
import requests
import dotenv
import re

# 📌 Load environment variables from `.env`
dotenv.load_dotenv()

# 📌 Secure API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:3000/chat")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing `GOOGLE_API_KEY` in `.env` file!")

# 📌 Configure Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# 📌 AI Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# 📌 Initialize the AI Model
model = genai.GenerativeModel(
    model_name="tunedModels/neuronestmodel2-nrg6xqft2zvd",
    generation_config=generation_config,
)

# 📌 Audio recording parameters
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16
SILENCE_THRESHOLD = 500  
SILENCE_DURATION = 1.0  
MAX_RECORDING_DURATION = 300  

# 📌 Initialize text-to-speech engine
engine = None  

# 📌 Function to save chat messages to backend
def save_chat(user_id, role, content):
    """Validate and send chat messages to backend"""
    if not content or len(content.strip()) == 0:
        print("❌ Cannot save empty message.")
        return

    if len(content) > 5000:  # Prevent excessively long messages
        print("❌ Message too long!")
        return

    payload = {"userId": user_id, "role": role, "content": content}

    try:
        response = requests.post(BACKEND_API_URL, json=payload)
        if response.status_code == 200:
            print("✅ Chat saved to database")
        else:
            print(f"❌ Error saving chat: {response.json()}")
    except Exception as e:
        print(f"⚠ Error sending chat to backend: {str(e)}")


# 📌 Function to stop speech output
def stop_speech():
    if engine:
        engine.stop()

# 📌 Function to convert text to speech
def text_to_speech(text):
    """Convert text to speech using pyttsx3"""
    print(f"🗣 AI says: {text}")
    
    global engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"⚠ Error in speech synthesis: {str(e)}")

# 📌 Function to remove markdown formatting
def clean_markdown(text):
    text = re.sub(r'[*_#>`\[\]]+', '', text)
    return text.strip()       

# 📌 Function to check for silence
def is_silent(audio_data):
    return np.max(np.abs(audio_data)) < SILENCE_THRESHOLD

def save_audio(recording, sample_rate, filename):
     """Saves recorded audio to a WAV file"""
     with wave.open(filename, 'wb') as wav_file:
         wav_file.setnchannels(1)
         wav_file.setsampwidth(2)
         wav_file.setframerate(sample_rate)
         wav_file.writeframes(recording.tobytes())

def speech_to_text(audio_file):
    """Convert speech to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition failed"

# 📌 Function to record audio
def record_audio():
    """Records audio and stops when silence is detected"""
    global audio_queue
    audio_queue = queue.Queue()
    recording = []
    silence_start = None
    start_time = time.time()
    
    print("🎤 Recording... Speak now!")

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        while True:
            try:
                audio_data = audio_queue.get(timeout=0.1)
                recording.append(audio_data)

                if is_silent(audio_data):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        print("⏳ Silence detected, stopping recording...")
                        break
                else:
                    silence_start = None
            except queue.Empty:
                continue

    return np.concatenate(recording, axis=0), SAMPLE_RATE

# 📌 Function to save audio
# def save_audio(recording, sample_rate, filename):
#     """Saves recorded audio to a WAV file"""
#     with wave.open(filename, 'wb') as wav_file:
#         wav_file.setnchannels(1)
#         wav_file.setsampwidth(2)
#         wav_file.setframerate(sample_rate)
#         wav_file.writeframes(recording.tobytes())

# # 📌 Function to convert speech to text
# def speech_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio = recognizer.record(source)
#         try:
#             return recognizer.recognize_google(audio)
#         except sr.UnknownValueError:
#             return "Could not understand audio"
#         except sr.RequestError:
#             return "Speech recognition failed"




# 📌 Function to speak farewell
def speak_farewell():
    """Says a farewell message"""
    farewell = "Thank you for talking to me. Goodbye!"
    print(farewell)
    text_to_speech(farewell)

# 📌 Main chat loop
def main():
    chat_session = model.start_chat()
    user_id = input("🆔 Enter your User ID: ")

    while True:
        try:
            print("\nChoose input method:")
            print("1. Voice input 🎤")
            print("2. Text input ⌨")
            print("3. Exit ❌")

            choice = input("Enter your choice (1-3): ").strip()

            if choice == "3":
                speak_farewell()
                os._exit(0)

            if choice == "1":
                recording, sample_rate = record_audio()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    save_audio(recording, sample_rate, temp_file.name)

                temp_filename = temp_file.name  # Store filename before using it

                try:
                    user_input = speech_to_text(temp_filename)
                    print(f"🗣 You said: {user_input}")
                finally:
                    time.sleep(0.5)  # Ensure file is closed before deletion
                    try:
                        os.remove(temp_filename)  # Delete file safely
                    except PermissionError:
                        print(f"⚠ Could not delete {temp_filename}. It may still be in use.")
                        continue

            elif choice == "2":
                user_input = input("\n✍ Enter your message: ").strip()

            if user_input.lower() in ['quit', 'exit', 'stop']:
                speak_farewell()
                os._exit(0)

            save_chat(user_id, "user", user_input)  # ✅ Store user message

            response = chat_session.send_message(user_input)
            clean_response = clean_markdown(response.text)
            print(f"🤖 AI response: {clean_response}")

            save_chat(user_id, "assistant", clean_response)  # ✅ Store AI response

            text_to_speech(clean_response)

        except Exception as e:
            print(f"⚠ Error: {str(e)}")
            sys.exit(1)

# 📌 Run chatbot
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
        os._exit(0)
    except Exception as e:
        print(f"⚠ Fatal error: {str(e)}")
        os._exit(1)

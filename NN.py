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
import msvcrt  # For Windows keyboard input
import re

# Configure API keys
genai.configure(api_key="AIzaSyCpJs-H99ZXRB5OvHLKXuarif6B6m7lz1Q")

# Audio recording parameters
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16
SILENCE_THRESHOLD = 500  # Adjust this value based on your microphone sensitivity
SILENCE_DURATION = 1.0  # Seconds of silence to trigger stop
MAX_RECORDING_DURATION = 30  # Maximum recording duration in seconds

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="tunedModels/neuronestmodel2-nrg6xqft2zvd",
    generation_config=generation_config,
)

def stop_speech():
    """Stop the current speech"""
    engine.stop()

def text_to_speech(text):
    """Convert text to speech using pyttsx3 with interruption capability"""
    try:
        print("Converting text to speech... (Press 's' to stop)")
        print(f"Text to convert: {text}")
        
        # Create a new engine instance for each speech
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        local_engine.setProperty('volume', 0.9)
        
        speech_stopped = False
        
        # Start speech in a separate thread
        def speak():
            try:
                if not speech_stopped:
                    local_engine.say(text)
                    local_engine.runAndWait()
            except Exception as e:
                print(f"Speech thread error: {str(e)}")
        
        speech_thread = threading.Thread(target=speak)
        speech_thread.daemon = True  # Make the thread a daemon so it exits when the main program exits
        speech_thread.start()
        
        # Monitor for stop key or speech completion
        while speech_thread.is_alive():
            if msvcrt.kbhit():  # Check if a key is pressed
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 's':
                    speech_stopped = True
                    try:
                        local_engine.stop()
                    except:
                        pass
                    print("\nSpeech stopped. Starting new recording...")
                    return True  # Indicate that speech was stopped
            
            # Small delay to prevent high CPU usage
            time.sleep(0.1)
        
        # Clean up
        try:
            del local_engine
        except:
            pass
            
        print("Speech completed")
        return False  # Indicate speech completed normally
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return False

def audio_callback(indata, frames, time, status):
    """Callback function for audio recording"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def is_silent(audio_data):
    """Check if the audio data is silent"""
    return np.max(np.abs(audio_data)) < SILENCE_THRESHOLD

def record_audio():
    """Record audio from microphone with voice activity detection"""
    global audio_queue
    audio_queue = queue.Queue()
    recording = []
    silence_start = None
    start_time = time.time()
    initial_delay = 1.0  # Reduced from 2.0 to 0.5 seconds
    speech_detected = False
    
    print("Recording... Speak now!")
    
    # Create and start the input stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype=DTYPE, 
        callback=audio_callback
    )
    
    with stream:
        stream.start()
        while True:
            try:
                # Get audio data from queue
                audio_data = audio_queue.get(timeout=0.1)
                recording.append(audio_data)
                
                # Wait for initial delay before any detection
                if time.time() - start_time <= initial_delay:
                    continue
                
                # Check if speech has started
                if not speech_detected and not is_silent(audio_data):
                    speech_detected = True
                    print("Speech detected, listening...")
                
                # Only check for silence after speech is detected
                if speech_detected:
                    if is_silent(audio_data):
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            print("Silence detected, stopping recording...")
                            break
                    else:
                        silence_start = None
                
                # Check maximum recording duration
                if time.time() - start_time >= MAX_RECORDING_DURATION:
                    print("Maximum recording duration reached")
                    break
                    
            except queue.Empty:
                continue
            
        # Properly stop and close the stream
        stream.stop()
        stream.close()
    
    # Clear the queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    
    return np.concatenate(recording, axis=0), SAMPLE_RATE

def save_audio(recording, sample_rate, filename):
    """Save recorded audio to WAV file in PCM format"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(recording.tobytes())

def speech_to_text(audio_file):
    """Convert speech to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"

def clean_markdown(text):
    """Convert markdown text to plain text by removing markdown formatting"""
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Bold (alternative)
    text = re.sub(r'_(.*?)_', r'\1', text)        # Italic (alternative)
    
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove lists
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove blockquotes
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    
    # Remove links but keep the text
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    
    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def main():
    chat_session = model.start_chat()
    conversation_history = []
    
    while True:
        try:
            # Ask user for input method
            print("\nChoose input method:")
            print("1. Voice input")
            print("2. Text input")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "3":
                farewell = "Thank you for talking to me. I hope I helped you to your content. Goodbye!"
                print(farewell)
                text_to_speech(farewell)
                print("Exiting program...")
                # Clean up and exit
                try:
                    sd.stop()
                except:
                    pass
                sys.exit(0)
            
            if choice == "1":
                # Voice input
                recording, sample_rate = record_audio()
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    save_audio(recording, sample_rate, temp_file.name)
                    
                    try:
                        # Convert speech to text
                        user_input = speech_to_text(temp_file.name)
                        print(f"You said: {user_input}")
                        
                        if user_input.lower() in ['quit', 'exit', 'stop']:
                            farewell = "Thank you for talking to me. I hope I helped you to your content. Goodbye!"
                            print(farewell)
                            text_to_speech(farewell)
                            print("Exiting program...")
                            # Clean up and exit
                            try:
                                sd.stop()
                            except:
                                pass
                            sys.exit(0)
                            
                        # Add user input to conversation history
                        conversation_history.append({"role": "user", "content": user_input})
                        
                        # Get model response with conversation history
                        response = chat_session.send_message(user_input)
                        
                        # Clean markdown formatting from response
                        clean_response = clean_markdown(response.text)
                        
                        # Print cleaned response
                        print(f"Model response: {clean_response}")
                        
                        # Add original response to conversation history
                        conversation_history.append({"role": "model", "content": response.text})
                        
                        # Convert cleaned response to speech and check if it was interrupted
                        was_interrupted = text_to_speech(clean_response)
                        
                        # If speech was not interrupted, automatically continue to next recording
                        if not was_interrupted:
                            print("\nListening for your next input...")
                        
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
            
            elif choice == "2":
                # Text input
                user_input = input("\nEnter your message: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    farewell = "Thank you for talking to me. I hope I helped you to your content. Goodbye!"
                    print(farewell)
                    text_to_speech(farewell)
                    print("Exiting program...")
                    # Clean up and exit
                    try:
                        sd.stop()
                    except:
                        pass
                    sys.exit(0)
                
                # Add user input to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Get model response with conversation history
                response = chat_session.send_message(user_input)
                
                # Clean markdown formatting from response
                clean_response = clean_markdown(response.text)
                
                # Print cleaned response
                print(f"Model response: {clean_response}")
                
                # Add original response to conversation history
                conversation_history.append({"role": "model", "content": response.text})
                
                # Convert cleaned response to speech and check if it was interrupted
                was_interrupted = text_to_speech(clean_response)
                
                # If speech was not interrupted, automatically continue to next input
                if not was_interrupted:
                    print("\nReady for next input...")
            
            else:
                print("Invalid choice. Please try again.")
                
        except SystemExit:
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
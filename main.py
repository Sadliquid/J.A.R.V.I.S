import pyaudio
import wave
import requests
import json
import tempfile
import os
import whisper

RECORD_SECONDS = 4
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"


def record_audio():
    print(f"Recording...")

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames

def save_audio(frames, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio_openai_whisper(audio_file):
    model = whisper.load_model("base")

    result = model.transcribe(audio_file, fp16=False)
    transcript = result["text"].strip()

    return transcript

def query_local_llm(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "No response received")

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running locally."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with LLM: {e}"
    except json.JSONDecodeError:
        return "Error: Invalid response from LLM"

def main():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_filename = temp_audio.name

    try:
        frames = record_audio()

        save_audio(frames, temp_filename)

        transcript = transcribe_audio_openai_whisper(temp_filename)

        if not transcript or transcript.strip() == "":
            print("No speech detected.")
            return

        if transcript.startswith("Error") or transcript.startswith("Could not"):
            print(f"Transcription failed: {transcript}")
            return

        llm_response = query_local_llm(transcript)

        print("-" * 50)
        print(f"Transcript: {transcript}")
        print(f"Response: {llm_response}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

if __name__ == "__main__":
    main()
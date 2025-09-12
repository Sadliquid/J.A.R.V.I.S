import pyaudio
import wave
import requests
import json
import tempfile
import os
import whisper
import socket
from datetime import datetime

RECORD_SECONDS = 4
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"
SEARCH_API_URL = "https://api.duckduckgo.com/"

def check_internet_connection():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def needs_web_search(prompt):
    print("\n🤖 Thinking...")
    search_check_prompt = f"""You are a search decision system. Analyze this user prompt and determine if it requires real-time web search for current/recent information that you cannot provide due to your knowledge cutoff.

User prompt: "{prompt}"

Consider these factors:
- Does it ask for current events, news, weather, stock prices, or real-time data?
- Does it ask for recent information (today, this week, latest, current, now)?
- Does it ask for live data like sports scores, market prices, weather or current conditions?
- Does it contain words like "today", "now", "yesterday", "current", "latest", "recent", "breaking", or any word that suggests an element of futurity or present moment?
- Does it require real-time information that you cannot provide unless you search the web?

Answer with "False" if the prompt can be answered based on your existing knowledge. Answer with "True" ONLY if web search is needed. You must answer with either "True" or "False" and nothing else.

Answer:"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": search_check_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip().lower()
        return "true" in answer
    except:
        return False

def search_web(query):
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }

        response = requests.get(SEARCH_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []

        if data.get('Abstract'):
            results.append(data['Abstract'])
        if data.get('Answer'):
            results.append(data['Answer'])
        if data.get('Definition'):
            results.append(data['Definition'])
        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(topic['Text'])

        return " ".join(results) if results else None

    except Exception as e:
        return None

def get_answer_with_search(prompt):
    search_results = search_web(prompt)
    if not search_results:
        return get_direct_answer(prompt)

    current_time = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')

    answer_prompt = f"""Current date and time: {current_time}

Search results: {search_results}

User question: {prompt}

Provide a short answer to the user's question using the search results and current time. Answer in the most concise way possible. Only use the search results if they are relevant to the question. If the search results are not relevant, answer based on your existing knowledge. If you are unable to answer, say "I'm not sure.". Do not make up or fabricate information. You must use the following examples as a guide for how to respond.

Examples:
- If asked "What day is it today?" answer "It's Friday, the 31st of December, 2023"
- If asked "What's the weather?" answer "It's currently sunny and 35°C"
- If asked "What time is it?" answer "It's 3:45 PM"

Answer:"""

    return query_ollama(answer_prompt)

def get_direct_answer(prompt):
    current_time = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')

    answer_prompt = f"""Current date and time: {current_time}

User question: {prompt}

Provide a short answer to the user's question. Answer in the most concise way possible. If you are unable to answer, say "I'm not sure.". Do not make up or fabricate information. You must use the following examples as a guide for how to respond.

Examples:
- If asked "What day is it today?" answer "It's Friday, the 31st of December, 2023"
- If asked "What's the weather?" answer "It's currently sunny and 35°C"
- If asked "What time is it?" answer "It's 3:45 PM"

Answer:"""

    return query_ollama(answer_prompt)

def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except json.JSONDecodeError:
        return "Error: Invalid response"

def record_audio():
    print(f"\nSpeak now.")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
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
    result = model.transcribe(audio_file, fp16=False, language="en")
    transcript = result["text"].strip()
    return transcript

def speak_text(text, voice="Daniel"):
    try:
        os.system(f'say -v "{voice}" "{text}"')
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def main():
    print("--------------------------------")
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

        if needs_web_search(transcript):
            if check_internet_connection():
                print("\n🔍 Searching the web...")
                answer = get_answer_with_search(transcript)
            else:
                answer = get_direct_answer(transcript)
        else:
            answer = get_direct_answer(transcript)

        print(f"\n{answer}")
        speak_text(answer)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

if __name__ == "__main__":
    main()
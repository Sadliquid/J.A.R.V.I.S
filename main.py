import pyaudio
import wave
import requests
import json
import tempfile
import os
import whisper
import socket
from datetime import datetime
import re

RECORD_SECONDS = 4
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"

SEARCH_API_URL = "https://api.duckduckgo.com/"
SEARCH_ENABLED = True

def check_internet_connection():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def needs_search(query):
    search_indicators = [
        # Time-related
        r'\b(today|now|current|latest|recent)\b',
        r'\b(weather|temperature)\b',
        r'\b(news|headlines)\b',
        r'\b(stock|price|market)\b',
        r'\b(what time|what day|date)\b',

        # Real-time data
        r'\b(happening now|breaking)\b',
        r'\b(live|real[- ]?time)\b',

        # Questions that typically need current data
        r'\b(who is the (current|new))\b',
        r'\b(what happened (today|yesterday|this week))\b',
        r'\b(sports scores?|game results?)\b',
        r'\b(exchange rate|currency)\b',

        # Specific current info requests
        r'\b(search for|look up|find)\b',
        r'\b(what\'s (new|happening))\b'
    ]

    query_lower = query.lower()
    for pattern in search_indicators:
        if re.search(pattern, query_lower):
            return True
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
            results.append(f"Summary: {data['Abstract']}")

        if data.get('Answer'):
            results.append(f"Direct answer: {data['Answer']}")

        if data.get('Definition'):
            results.append(f"Definition: {data['Definition']}")

        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:2]:  # Limit to 2
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(f"Related: {topic['Text']}")

        return results if results else ["No specific information found"]

    except Exception as e:
        return [f"Search error: {str(e)}"]

def get_current_datetime():
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

def enhance_prompt_with_context(original_prompt):
    enhanced_prompt = original_prompt
    context_added = []

    if re.search(r'\b(time|date|day|today|now|yesterday|tomorrow|tonight|later|ago)\b', original_prompt.lower()):
        datetime_info = get_current_datetime()
        context_added.append(datetime_info)

    if needs_search(original_prompt):
        if check_internet_connection():
            print("🔍 Searching the web...")
            search_results = search_web(original_prompt)

            if search_results and not any("error" in result.lower() for result in search_results):
                context_added.append("Recent search results:")
                context_added.extend(search_results)
            else:
                context_added.append("Note: Search attempted but no reliable results found.")
        else:
            print("⚠️ Internet search needed but no connection available.")
            context_added.append("Note: This query may benefit from current information, but internet is unavailable.")

    if context_added:
        context_section = "\n".join(context_added)
        enhanced_prompt = f"""Context Information:
{context_section}

User Question: {original_prompt}

Please provide a super short and concise answer using the information above along with your knowledge. If the context does not help, answer based on your knowledge. Keep answers as short as possible."""

    return enhanced_prompt

def record_audio():
    print(f"\nSpeak now.")
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

        enhanced_prompt = enhance_prompt_with_context(transcript)

        llm_response = query_local_llm(enhanced_prompt)

        print(f"\n{llm_response}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

if __name__ == "__main__":
    main()
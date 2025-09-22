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
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search.json"
MODE = os.environ.get("MODE")

def check_internet_connection():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def needs_web_search(prompt):
    print("\n🤖 Thinking...")
    search_check_prompt = f"""
        1.) You are a strict yes/no system.

        2.) If the question is about current date or time, answer "False", and ignore points 3 and 4.

        3.) If the question can be answered with built-in knowledge, answer "False".

        4.) Only answer "True" if the question clearly requires current, real-time, or recent information, such as news, weather, recent events, or anything that cannot be answered with static knowledge.

        Question: "{prompt}"
        Answer:
    """

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

def search_web_with_serp(query):
    print("\n🔍 Searching the web with SerpAPI...")
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY
        }
        response = requests.get(SERPAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []

        # Pull snippets from organic search results
        for result in data.get("organic_results", [])[:3]:
            if "snippet" in result:
                results.append(result["snippet"])

        # Save search results to file
        if results:
            save_search_results(query, results, data, engine="SerpAPI")

        return " ".join(results) if results else None
    except Exception as e:
        return None

def search_web_with_ddg(query):
    print("\n🔍 Searching the web with DuckDuckGo...")
    try:
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(SEARCH_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []

        # Pull snippets from related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if "Text" in topic:
                results.append(topic["Text"])

        # Save search results to file
        if results:
            save_search_results(query, results, data, engine="DuckDuckGo")

        return " ".join(results) if results else None
    except Exception:
        return None

def save_search_results(query, results, full_data, engine):
    """Save search results to a text file with timestamp and query information."""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = "search_results.txt"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Search Query: {query}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Engine: {engine}\n")
            f.write(f"{'='*60}\n")

            # Write the search result snippets
            for i, result in enumerate(results, 1):
                f.write(f"\nResult {i}:\n{result}\n")

            # Write additional details from full search data
            organic_results = full_data.get("organic_results", [])[:3]
            if organic_results:
                f.write(f"\nDetailed Results:\n")
                for i, result in enumerate(organic_results, 1):
                    f.write(f"\n{i}. Title: {result.get('title', 'N/A')}\n")
                    f.write(f"   URL: {result.get('link', 'N/A')}\n")
                    f.write(f"   Snippet: {result.get('snippet', 'N/A')}\n")

            f.write(f"\n{'='*60}\n")

    except Exception as e:
        print(f"Error saving search results: {e}")

def get_answer_with_search(prompt):
    search_results = search_web_with_serp(prompt) if MODE == "Online" else search_web_with_ddg(prompt)
    if not search_results:
        return get_direct_answer(prompt)

    now = datetime.now()
    current_date = now.strftime('%A, %B %d, %Y')
    current_time = now.strftime('%I:%M %p')

    answer_prompt = f"""
        Today's date: {current_date}.

        Current time: {current_time}.

        Relevant information: {search_results}

        User question: {prompt}

        Answer the question with ONLY the final response.
        Always phrase the answer as a complete, natural sentence.
        Do not just copy numbers or fragments from the information.
        Do not include any preamble, explanations, or extra words.
        Do not use phrases like "according to" or "based on".
        Do not say things like "Sure," or "Here is my answer."
        If you do not know how to answer, say "I'm not sure."

        Examples (MUST FOLLOW):
        - "What day is it today?" → "It's [DAY], the [DATE] of [MONTH], [YEAR]."
        - "What time is it?" → "It's [TIME] [AM/PM]."

        Answer:
    """

    return query_ollama(answer_prompt)

def get_direct_answer(prompt):
    now = datetime.now()
    current_date = now.strftime('%A, %B %d, %Y')
    current_time = now.strftime('%I:%M %p')

    answer_prompt = f"""
        Today's date: {current_date}.

        Current time: {current_time}.

        User question: {prompt}

        Always phrase the answer as a complete, natural sentence.
        Do not include any preamble, explanations, or extra words.
        Do not use phrases like "according to" or "based on".
        Do not say things like "Sure," or "Here is my answer."
        If you do not know how to answer, say "I'm not sure.".
        Do not answer the question if it requires current, real-time, or recent information. Instead, say "I'm not sure.".
        Respond exactly as in the examples below:

        Examples (MUST FOLLOW):
        - "What day is it today?" → "It's [DAY], the [DATE] of [MONTH], [YEAR]."
        - "What time is it?" → "It's [TIME] [AM/PM]."

        Answer:
    """

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
import asyncio
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

client = OpenAI()
async_client = AsyncOpenAI()

SYSTEM_PROMPT = """
You are an expert voice assistant.
Always respond in a delightful, friendly, and happy tone.
Your response will be converted to speech and played back to the user.
"""

# ---------- AUDIO PLAYER ----------
class LocalAudioPlayer:
    async def play(self, response):
        pcm_chunks = []

        async for event in response:
            if event.type == "response.audio.delta":
                pcm_chunks.append(event.delta)

        audio = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)
        sd.play(audio, samplerate=24000)
        sd.wait()

# ---------- TEXT TO SPEECH ----------
async def tts(text: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="Speak happily and delightfully",
        input=text,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

# ---------- MAIN LOOP ----------
def main():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)

        print("🎙️ Voice agent started (Ctrl+C to stop)")

        while True:
            print("\nSpeak something...")
            audio = recognizer.listen(source)

            try:
                user_text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print("Could not understand audio")
                continue

            print("You:", user_text)
            messages.append({"role": "user", "content": user_text})

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=messages
            )

            ai_text = response.output_text
            print("AI:", ai_text)

            messages.append({"role": "assistant", "content": ai_text})

            asyncio.run(tts(ai_text))


if __name__ == "__main__":
    main()

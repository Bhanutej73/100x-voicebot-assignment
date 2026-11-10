import asyncio
import hashlib
import io
import os
import re
from pathlib import Path

import streamlit as st
from groq import Groq

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------- Config ----------
LLM_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "qwen/qwen3.8-27b"
STT_MODEL = "whisper-large-v3-turbo"
TTS_VOICE = "en-IN-PrabhatNeural"  # change to any edge-tts voice
MAX_HISTORY = 10  # last N messages sent to the model
SYSTEM_PROMPT = Path("prompt.txt").read_text(encoding="utf-8")

SAMPLE_QUESTIONS = [
    "What should we know about your life story?",
    "What's your #1 superpower?",
    "What are the top 3 areas you'd like to grow in?",
    "What misconception do your coworkers have about you?",
    "How do you push your boundaries and limits?",
]

st.set_page_config(page_title="Talk to Bhanu", page_icon="🎙️", layout="centered")


@st.cache_resource
def get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        st.error("Server is missing GROQ_API_KEY. Add it in the Space secrets.")
        st.stop()
    return Groq(api_key=key)


# ---------- Pipeline ----------
def transcribe(audio_bytes: bytes) -> str:
    result = get_client().audio.transcriptions.create(
        file=("speech.wav", audio_bytes),
        model=STT_MODEL,
        language="en",
        temperature=0.0,
    )
    return result.text.strip()


def ask_llm(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]
    last_err = None
    for model in (LLM_MODEL, FALLBACK_MODEL):
        try:
            kwargs = dict(model=model, messages=messages, temperature=0.7, max_tokens=500)
            if model == LLM_MODEL:
                kwargs["extra_body"] = {"reasoning_effort": "low"}  # lower latency
            resp = get_client().chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:  # rate limit, model hiccup -> try fallback
            last_err = e
    raise RuntimeError(f"LLM failed: {last_err}")


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize(text: str) -> bytes | None:
    text = clean_for_speech(text)
    try:
        import edge_tts

        async def _run() -> bytes:
            buf = io.BytesIO()
            async for chunk in edge_tts.Communicate(text, TTS_VOICE).stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        audio = asyncio.run(_run())
        if audio:
            return audio
    except Exception:
        pass
    try:  # fallback
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text, lang="en", tld="co.in").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def handle_user_turn(user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    try:
        reply = ask_llm(st.session_state.messages)
    except Exception:
        st.session_state.messages.pop()
        st.error("I'm a bit overloaded right now. Please try again in a few seconds.")
        return
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.pending_audio = synthesize(reply)


# ---------- UI ----------
st.title("🎙️ Talk to Bhanu")
st.caption(
    "An AI voice bot that answers interview questions in my voice and from my "
    "real background. Tap the mic, ask a question, and listen."
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_audio", None)
st.session_state.setdefault("last_audio_hash", None)

audio = st.audio_input("Tap to record, tap again to stop")
if audio is not None:
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if digest != st.session_state.last_audio_hash:  # avoid reprocessing on rerun
        st.session_state.last_audio_hash = digest
        with st.spinner("Listening..."):
            try:
                question = transcribe(data)
            except Exception:
                question = ""
                st.error("Couldn't transcribe that. Please try again.")
        if question:
            with st.spinner("Thinking..."):
                handle_user_turn(question)

st.write("**Or try a sample question:**")
for q in SAMPLE_QUESTIONS:
    if st.button(q, use_container_width=True):
        with st.spinner("Thinking..."):
            handle_user_turn(q)

typed = st.chat_input("...or type your question")
if typed:
    with st.spinner("Thinking..."):
        handle_user_turn(typed)

st.divider()
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["content"])

if st.session_state.pending_audio:  # play once, only on the run that created it
    st.audio(st.session_state.pending_audio, format="audio/mp3", autoplay=True)
    st.session_state.pending_audio = None
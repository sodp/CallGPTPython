from services.stt_tts_seamless_m4t import *
from services.stt_whisper import *
from pydantic import BaseModel
import asyncio

class AudioTranscriptionRequest(BaseModel):
    audio_array: list
    sample_rate: int = 16000

class TextToMulawRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

# Cache dictionary to store results of text-to-speech requests
text_to_speech_cache = {}
# Asynchronously execute CPU-bound tasks
async def async_text_to_speech(text, src_lang, tgt_lang):
    return await asyncio.to_thread(text_to_speech, text, src_lang, tgt_lang)

async def async_convert_to_mulaw(audio_array, sample_rate):
    return await asyncio.to_thread(convert_to_mulaw, audio_array, sample_rate)

async def async_transcribe_audio_seamless_m4t(audio_array, sample_rate):
    return await asyncio.to_thread(transcribe_audio_seamless_m4t, audio_array, sample_rate)

async def async_transcribe_audio_whisper(audio_array, sample_rate):
    return await asyncio.to_thread(transcribe_audio_whisper, audio_array, sample_rate)

# async def async_transcribe_audio_whisper_large_v3(audio_array, sample_rate):
#     return await asyncio.to_thread(transcribe_audio_whisper_large_v3, audio_array, sample_rate)

# Function to handle text-to-speech requests and cache results
async def cached_text_to_speech(text, src_lang, tgt_lang):
    key = (text, src_lang, tgt_lang)
    if key not in text_to_speech_cache:
        text_to_speech_cache[key] = await async_text_to_speech(text, src_lang, tgt_lang)
    return text_to_speech_cache[key]
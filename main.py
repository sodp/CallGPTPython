from fastapi import FastAPI, HTTPException
from services.cache import *
import logging
import torch


app = FastAPI()
torch.cuda.empty_cache()
logging.basicConfig(filename='stt_tts.log', level=logging.INFO)


@app.post("/text-to-mulaw")
async def text_to_mulaw(request_body: TextToMulawRequest):
    text = request_body.text
    src_lang = request_body.src_lang
    tgt_lang = request_body.tgt_lang
    try:
        logging.info(f"Received payload: text={text}, src_lang={src_lang}, tgt_lang={tgt_lang}")
        audio_array = await cached_text_to_speech(text, src_lang, tgt_lang)
        sample_rate = seamlessM4T_model.config.sampling_rate
        mulaw_audio_array = await async_convert_to_mulaw(audio_array, sample_rate)
        return {"mulaw_audio": mulaw_audio_array.tolist(), "sample_rate": sample_rate}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post("/transcribe-audio-seamless-m4t")
async def transcribe_audio(request_body: AudioTranscriptionRequest):
    audio_array = request_body.audio_array
    sample_rate = request_body.sample_rate
    try:
        logging.info("Received audio for transcription")
        transcribed_text = await async_transcribe_audio_seamless_m4t(audio_array, sample_rate)
        return {"transcribed_text": transcribed_text}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/transcribe-audio-whisper")
async def transcribe_audio(request_body: AudioTranscriptionRequest):
    audio_array = request_body.audio_array
    sample_rate = request_body.sample_rate
    try:
        logging.info("Received audio for transcription")
        transcribed_text = await transcribe_audio_whisper(audio_array, sample_rate)
        return {"transcribed_text": transcribed_text}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    


# @app.post("/transcribe-audio-whisper-largw")
# async def speech_to_text(request_body: AudioTranscriptionRequest):
#     audio_array = request_body.audio_array
#     sample_rate = request_body.sample_rate
#     try:
#         logging.info("Received audio for transcription")
#         transcribed_text = async_transcribe_audio_whisper_large_v3(audio_array, sample_rate)
#         return {"text": transcribed_text}
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

    



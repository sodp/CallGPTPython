from transformers import WhisperProcessor, WhisperForConditionalGeneration , AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from typing import List

# Load model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
whisper_model.config.forced_decoder_ids = None
device = "cuda:0" 

def transcribe_audio_whisper(audio_array, sampling_rate):
    try:
        audio_tensor = torch.tensor(audio_array)
        input_features = whisper_processor(
            audio_tensor, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)
        # Generate token ids
        predicted_ids = whisper_model.generate(input_features)
        # Decode token ids to text
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription
    except Exception as e:
        raise RuntimeError(f"An error occurred during transcription: {str(e)}")


# v3 Large Whisper
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# model_id_whisper_large = "openai/whisper-large-v3"
# model_whisper_large = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id_whisper_large, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model_whisper_large.to(device)
# processor = AutoProcessor.from_pretrained(model_id_whisper_large)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model_whisper_large,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )


# def transcribe_audio_whisper_large_v3(audio: List[float], sampling_rate: int) -> str:
#     result = pipe({"array": audio, "sampling_rate": sampling_rate})
#     return result["text"]

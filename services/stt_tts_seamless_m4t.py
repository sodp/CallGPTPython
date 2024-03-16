import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel

device = "cuda:0" 
seamlessM4T_processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
seamlessM4T_model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
seamlessM4T_model.to(device)

def text_to_speech(input_text, src_lang="eng", tgt_lang="eng"):
    text_inputs = seamlessM4T_processor(text=input_text, src_lang=src_lang, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_tensor = seamlessM4T_model.generate(**text_inputs, tgt_lang=tgt_lang)[0]
        audio_array = audio_tensor.cpu().numpy().squeeze()  # Move tensor to CPU before converting to numpy
    return audio_array

def convert_to_mulaw(audio_array, sample_rate):
    audio_tensor = torch.tensor(audio_array).to(device)  # Ensure tensor is on the same device
    mu_law_transform = torchaudio.transforms.MuLawEncoding()
    mulaw_tensor = mu_law_transform(audio_tensor)
    mulaw_array = mulaw_tensor.cpu().numpy()  # Move tensor to CPU before converting to numpy
    return mulaw_array


def transcribe_audio_seamless_m4t(audio_array, sample_rate):
    audio_inputs = seamlessM4T_processor(audios=torch.tensor([audio_array]), return_tensors="pt").to(device)
    output_tokens = seamlessM4T_model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
    transcribed_text_from_audio = seamlessM4T_processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return transcribed_text_from_audio
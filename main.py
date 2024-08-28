import torch
from TTS.api import TTS
import gradio as gr

print(torch.cuda.is_available())

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" /
device = "mps"

def generate_audio(text="A journey of a thousand miles begins with a single step"):
    tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch').to(device)
    #tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v1.1')
    tts.tts_to_file(text=text, file_path="outputs/output.wav")
    return "outputs/output.wav"

print(generate_audio())



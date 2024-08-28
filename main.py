import torch
from TTS.api import TTS
import gradio as gr

print(torch.cuda.is_available())

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" /
device = "mps"

def generate_audio(text="A journey of a thousand miles begins with a single step"):
    #tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch').to(device)
    #tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v1.1', language_data='ar')
    tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2')
    tts.tts_to_file(text=text,
                    file_path="outputs/output.wav",
                    speaker="Abrahan Mack",
                    language="arab")
    return "outputs/output.wav"

print(generate_audio())


demo = gr.Interface(fn=generate_audio,
                    inputs=[gr.Text(label="text"),],
                    outputs=[gr.Audio(label="Audio")])

demo.launch()


#  > Available speaker ids: (Set --speaker_idx flag to one of these values to use the multi-speaker model.
# dict_keys(['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence',
# 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen',
# 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler',
# 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black',
# 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid',
# 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland',
# 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa',
# 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl',
# 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski'])

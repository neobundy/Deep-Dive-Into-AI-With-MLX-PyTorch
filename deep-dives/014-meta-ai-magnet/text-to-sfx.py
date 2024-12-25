from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained("facebook/audio-magnet-medium")

descriptions = ["gunfight under heavy rain", "explosion in a cave", "alien spaceship landing"]
model.set_generation_params(use_sampling=True, top_k=0, top_p=0.9, temperature=2.0, max_cfg_coef=10.0, min_cfg_coef=1.0, decoding_steps=[20, 10, 10, 10], span_arrangement='nonoverlap')
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")

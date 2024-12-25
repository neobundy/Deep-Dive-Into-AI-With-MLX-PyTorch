import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

prompt = "a portrait of a beautiful lady wearing glasses"
image = pipe(prompt).images[0]
image.save("portrait.png")
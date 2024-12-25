import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model and tokenizer from Hugging Face
# torch.float32 is used to avoid an error when loading the model on CPU: RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
# The current version of PyTorch does not support layer normalization on CPU for half-precision floating point numbers.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# The state_dict of a model contains the parameters of the model. Printing
# the state_dict's keys can help you understand the structure of the model.
# If you want to see the full detail, you may want to convert it to a dictionary and print that.
# However, this could be very verbose for large models.
print(model.state_dict().keys())
model_state_dict = model.state_dict()

prompt = "What is the sentiment of this text: I love my wife!"

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
response = tokenizer.batch_decode(outputs)[0]
print(response)
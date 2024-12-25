from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Prepare the input
input_text = "The capital of France is [MASK]."
inputs = tokenizer(input_text, return_tensors="pt")

# Predict the masked word
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Decode the predicted word
predicted_index = torch.argmax(predictions[0, inputs['input_ids'][0] == tokenizer.mask_token_id])
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(f"Predicted word: {predicted_token}")
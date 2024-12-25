from transformers import AutoModel, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example text input
text = "What is AI?"
encoded_input = tokenizer(text, return_tensors='pt')

# Model inference
output = model(**encoded_input)
print(output)
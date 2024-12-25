from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

def sentiment_analysis(model_name, text):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize the input text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Predict sentiment using the model
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Assuming the model has labels in this order: [negative, neutral, positive]
    labels = ['negative', 'neutral', 'positive']
    sentiment = labels[predictions.argmax().item()]

    return sentiment

# Example text
text = "I love my wife"

# List of model names
model_names = ["bert-base-uncased", "distilbert-base-uncased", "roberta-large"]

# Perform sentiment analysis with each model
for model_name in model_names:
    print(f"Model: {model_name}")
    print(f"Sentiment: {sentiment_analysis(model_name, text)}\n")

# Load a pre-trained and fine-tuned model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

# Analyze sentiment
result = sentiment_analyzer(text)

print(result)
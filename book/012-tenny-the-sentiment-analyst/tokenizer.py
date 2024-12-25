from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example sentences
sentence = "Hello, how are you doing today?"
korean_sentence = "안녕? 오늘 어떻게 지냈어?"

# Perform tokenization
tokens = tokenizer.tokenize(sentence)
korean_tokens = tokenizer.tokenize(korean_sentence)

print("English Tokens:", tokens)
print("Korean Tokens:", korean_tokens)
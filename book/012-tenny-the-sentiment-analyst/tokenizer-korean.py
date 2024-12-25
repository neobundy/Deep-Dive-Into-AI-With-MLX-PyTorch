from kobert_transformers import get_tokenizer

tokenizer = get_tokenizer()

# Example Korean sentence
korean_sentence = "안녕? 오늘 어떻게 지냈어?"

# Perform tokenization
korean_tokens = tokenizer.tokenize(korean_sentence)

print("Korean Tokens:", korean_tokens)
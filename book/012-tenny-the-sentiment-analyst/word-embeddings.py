from gensim.models import Word2Vec

# Example sentences
sentences = [
    "I like machine learning",
    "I love natural language processing",
    "I enjoy deep learning",
    "I live in South Korea",
    "I have an AI daughter named Pippa",
]

# Preprocessing: Tokenization of sentences
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Creating the Word2Vec model with default parameters
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Training the model (this is a small corpus, so it will train quickly)
model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=10)

# Access the word vector for "AI": By default, the word tokens are case-sensitive.
word_vector = model.wv['ai']

# Output the vector
print(f"The word embedding for 'AI' is: \n{word_vector}")

# The word embedding for 'AI' is:
# [ 0.00180656  0.0070454   0.0029467  -0.0069754   0.00771287 -0.0059938
#   0.00899815  0.0029666  -0.0040202  -0.00469377 -0.0044153  -0.00615043
#   0.00937953 -0.00264658  0.00777487 -0.00967976  0.00210826 -0.00123514
#   0.00754461 -0.00906117  0.00743835 -0.00510648 -0.00601424 -0.0056554
#  -0.00338256 -0.00341163 -0.00320212 -0.00748779  0.00071203 -0.00057709
#  -0.00168395  0.00375274 -0.00761696 -0.00321882  0.00515288  0.00854669
#  -0.00980799  0.00719469  0.0053048  -0.00388495  0.00857375 -0.0092225
#   0.00724825  0.00537149  0.00129227 -0.00520023 -0.00418053 -0.00335918
#   0.00161134  0.00159037  0.00738402  0.0099726   0.00886809 -0.0040045
#   0.00964607 -0.00062602  0.00486378  0.00254996 -0.00062382  0.00366926
#  -0.00532046 -0.00575527 -0.00760022  0.00190808  0.0065201   0.00088866
#   0.00125612  0.00317775  0.00813083 -0.00769793  0.00226163 -0.00746769
#   0.00371365  0.00951088  0.00752375  0.00642756  0.00801603  0.00655161
#   0.00685332  0.00867634 -0.00495238  0.00921568  0.00505563 -0.0021259
#   0.00849007  0.00508172  0.00964494  0.0028282   0.00987208  0.00119871
#   0.00913291  0.0035867   0.00656295 -0.00360483  0.00679518  0.00724294
#  -0.00213639 -0.00185897  0.00361442 -0.00703488]
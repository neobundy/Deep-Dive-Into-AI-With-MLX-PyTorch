import regex as re

text = "The quick, brown fox jumps over the lazy dog."

# Advanced regex search using the regex package
# Finding words that start with a vowel and end with a consonant, case-insensitive
words = re.findall(r'\b[AEIOUaeiou][a-z]*[bcdfghjklmnpqrstvwxyz]\b', text)

print(f"Words found: {words}")
# ['over']
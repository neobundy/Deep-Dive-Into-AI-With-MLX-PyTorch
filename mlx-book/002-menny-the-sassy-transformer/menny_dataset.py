import random
import string

CONSONANTS = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
VOWELS = 'aeiouAEIOU'
LETTERS = string.ascii_letters


def generate_dataset(num_samples):
    query_list = [''.join(random.choices(LETTERS, k=random.randint(1, 5))) for _ in range(num_samples)]
    label_list = [''.join([random.choice(VOWELS.upper() if char.islower() else VOWELS.lower()) if char.isalpha() else random.choice(CONSONANTS) for char in query]) for query in query_list]

    return query_list, label_list

# Generate 100 data points
query_list, label_list = generate_dataset(100)

print(query_list[:10])  # Display the first 10 samples
print(label_list[:10])  # Display the first 10 labels
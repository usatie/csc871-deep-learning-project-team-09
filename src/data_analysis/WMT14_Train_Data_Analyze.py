import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
import os

file_path = "./src/data_analyse/wmt14_translate_de-en_train.csv"
tokenizer_path = "./src/data_analyse/bpe_tokenizer.json"
vocab_size = 37000
chunk_size = 10000




if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    print(f"Tokenizer file does not exist at {tokenizer_path}")


# 1. Train BPE tokenizer on part of your corpus (do this once and reuse)
def train_bpe_tokenizer(sentences, vocab_size=37000, save_path=tokenizer_path):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>"])
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(save_path)
    return tokenizer


# Step 2: Load tokenizer if it exists, else train it
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Tokenizer loaded successfully.")
else:
    print(f"Tokenizer file does not exist at {tokenizer_path}. Training a new tokenizer...")

    # Read the data and create a list of sentences for both German and English
    german_sentences = []
    english_sentences = []
    chunk_iter = pd.read_csv(file_path, sep=',', header=None, chunksize=chunk_size, on_bad_lines='skip',
                             engine='python')

    for chunk in chunk_iter:
        for _, row in chunk.iterrows():
            german_sentences.append(str(row[0]))  # German sentence
            english_sentences.append(str(row[1]))  # English sentence

    # Combine both languages for training the tokenizer
    all_sentences = german_sentences + english_sentences

    # Train BPE tokenizer on the combined sentences
    tokenizer = train_bpe_tokenizer(all_sentences, vocab_size, tokenizer_path)
    print(f"Tokenizer trained and saved to {tokenizer_path}")








# Initialize a list to store sentence lengths
german_lengths = []
english_lengths = []

# Read the CSV in chunks

chunk_iter = pd.read_csv(file_path, sep=',', header=None, chunksize=chunk_size, on_bad_lines='skip', engine='python')

# Iterate over chunks
for chunk in chunk_iter:
    for _, row in chunk.iterrows():
        german_sentence = row[0]
        english_sentence = row[1]

        german_token_length = len(tokenizer.encode(str(german_sentence)).ids) if isinstance(german_sentence, str) else 0
        english_token_length = len(tokenizer.encode(str(english_sentence)).ids) if isinstance(english_sentence,
                                                                                              str) else 0

        german_lengths.append(german_token_length)
        english_lengths.append(english_token_length)

# Convert to DataFrame
data = pd.DataFrame({'German': german_lengths, 'English': english_lengths})

print(f"Total sentence pairs: {len(data)}")

# Print statistics for German sentence lengths
print("German Sentence Length Statistics:")
print(f"Mean:   {data['German'].mean():.2f}")
print(f"Median: {data['German'].median():.2f}")
print(f"Std:    {data['German'].std():.2f}\n")

# Print statistics for English sentence lengths
print("English Sentence Length Statistics:")
print(f"Mean:   {data['English'].mean():.2f}")
print(f"Median: {data['English'].median():.2f}")
print(f"Std:    {data['English'].std():.2f}")


# Plot histograms
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(data['German'], bins=2400, color='red', edgecolor='black')
plt.title('German Token Length Distribution')
plt.xlabel('Length (tokens)')
plt.ylabel('Number of Sentences')
plt.grid(False)
plt.xlim(0, 100)  # Limit x-axis

plt.subplot(1, 2, 2)
plt.hist(data['English'], bins=2400, color='blue', edgecolor='black')
plt.title('English Token Length Distribution')
plt.xlabel('Length (tokens)')
plt.ylabel('Number of Sentences')
plt.grid(False)
plt.xlim(0, 100)  # Limit x-axis

plt.tight_layout()
plt.savefig("./src/data_analyse/sentence_length_distribution_WMT14.png", dpi=300)
plt.show()

# Print statistics for German sentence lengths
print("German Sentence Length Statistics:")
print(f"Mean:   {data['German'].mean():.2f}")
print(f"Median: {data['German'].median():.2f}")
print(f"Std:    {data['German'].std():.2f}\n")

# Print statistics for English sentence lengths
print("English Sentence Length Statistics:")
print(f"Mean:   {data['English'].mean():.2f}")
print(f"Median: {data['English'].median():.2f}")
print(f"Std:    {data['English'].std():.2f}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, median, stdev
from tqdm import tqdm
tqdm.pandas()  # <-- This enables progress_apply


import spacy

spacy_zh = spacy.load("zh_core_web_sm")
spacy_en = spacy.load("en_core_web_sm")


# Cell 2: Define the path to your file
file_path = "./src/data_analysis/Sentence pairs in Mandarin Chinese-English - 2025-04-27.tsv"





# Cell 3: Load the TSV file into a DataFrame
df = pd.read_csv(file_path, sep='\t', header=None)


df.columns = ['id1', 'Chinese', 'id2', 'English']


# Cell 4: Display the first few rows to understand the format
df.head(10)


# Cell 5: Display summary
# print(df.head(10))
print(f"Total sentence pairs: {len(df)}")

# Define safe token length functions
def get_chinese_token_length(text):
    if isinstance(text, str):
        return len([token for token in spacy_zh(text)])
    return 0

def get_english_token_length(text):
    if isinstance(text, str):
        return len([token for token in spacy_en(text)])
    return 0


# Apply with progress bar
df['Chinese_token_length'] = df['Chinese'].progress_apply(get_chinese_token_length)
df['English_token_length'] = df['English'].progress_apply(get_english_token_length)

chinese_mean = df['Chinese_token_length'].mean()
chinese_median = df['Chinese_token_length'].median()
chinese_std = df['Chinese_token_length'].std()

# Compute statistics for English word lengths
english_mean = df['English_token_length'].mean()
english_median = df['English_token_length'].median()
english_std = df['English_token_length'].std()


# Print the results
print("Chinese token Lengths:")
print(f"  Mean: {chinese_mean:.2f}")
print(f"  Median: {chinese_median}")
print(f"  Standard Deviation: {chinese_std:.2f}")

print("\nEnglish token Lengths:")
print(f"  Mean: {english_mean:.2f}")
print(f"  Median: {english_median}")
print(f"  Standard Deviation: {english_std:.2f}")

# Cell 6: Plot histograms
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(df['Chinese_token_length'], bins=300, color='red',edgecolor='black')

plt.title('Chinese Sentence Length Distribution')

plt.xlabel('Length (tokens)')
plt.ylabel('Number of Sentences')
plt.grid(False)
plt.xlim(0, 100)  # Limit x-axis

plt.subplot(1, 2, 2)
plt.hist(df['English_token_length'], bins=300, color='blue', edgecolor='black')
plt.title('English Sentence Length Distribution')
plt.xlabel('Length (tokens)')
plt.ylabel('Number of Sentences')
plt.grid(False)
plt.xlim(0, 60)  # Limit x-axis

plt.tight_layout()

plt.savefig("sentence_length_distribution_Tatoeba", dpi=300)

plt.show()
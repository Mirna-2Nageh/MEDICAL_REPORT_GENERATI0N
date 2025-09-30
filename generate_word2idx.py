import pandas as pd
import nltk
from collections import Counter
import pickle
import re

# Download NLTK data
nltk.download('punkt')

def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z.,]", " ", s)
    s = re.sub(r"\.+", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_vocabulary(reports, max_vocab_size=1806):  # 1806 + 4 special tokens = 1810
    text = " ".join(reports)
    tokens = nltk.word_tokenize(text)
    counter = Counter(tokens)
    # Get the most common words up to max_vocab_size
    most_common = counter.most_common(max_vocab_size)
    word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    j = 4
    for word, _ in most_common:
        word2idx[word] = j
        j += 1
    return word2idx

# Load dataset
try:
    df = pd.read_csv("indiana_reports.csv")
    df.dropna(subset=['findings', 'impression'], inplace=True)
    df['report'] = df['findings'] + ' ' + df['impression']
    df['report'] = df['report'].apply(clean_text)
    word2idx = build_vocabulary(df['report'])
    with open("word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)
    print(f"Generated word2idx.pkl with {len(word2idx)} tokens.")
    print(f"Special tokens: <PAD>={word2idx.get('<PAD>')}, <UNK>={word2idx.get('<UNK>')}, "
          f"<SOS>={word2idx.get('<SOS>')}, <EOS>={word2idx.get('<EOS>')}")
except FileNotFoundError:
    print("Error: indiana_reports.csv not found. Please ensure it is in the current directory.")
except Exception as e:
    print(f"Error generating vocabulary: {e}")
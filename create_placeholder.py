import pickle

# IMPORTANT: These indices (0, 1, 2, 3) MUST match what your model expects 
# for <pad>, <end>, <start>, and <unk> tokens.
placeholder_vocab = {
    '<pad>': 0,
    '<end>': 1,
    '<start>': 2,
    '<unk>': 3,
    # Add a few common words used in medical reports if you can recall them
    'lungs': 4,
    'clear': 5,
    'no': 6,
    'evidence': 7,
    'of': 8,
    'acute': 9,
    'disease': 10
}

# Save the dictionary as word2idx.pkl
try:
    with open('word2idx.pkl', 'wb') as f:
        pickle.dump(placeholder_vocab, f)
    print("Successfully created 'word2idx.pkl' placeholder file.")
except Exception as e:
    print(f"Error creating file: {e}")

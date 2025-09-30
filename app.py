import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import nltk
from collections import Counter
import math
import pickle

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Function to validate and fix word2idx
def ensure_special_tokens(word2idx):
    special_tokens = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SOS>": 2,
        "<EOS>": 3
    }
    # Check if special tokens are present
    missing_tokens = [token for token in special_tokens if token not in word2idx]
    if missing_tokens:
        st.warning(f"Missing special tokens {missing_tokens} in word2idx. Adding them.")
        # Create a new dictionary with special tokens
        new_word2idx = special_tokens.copy()
        # Reassign existing words with new indices starting from 4
        next_idx = 4
        for word, idx in word2idx.items():
            if word not in special_tokens:
                new_word2idx[word] = next_idx
                next_idx += 1
        return new_word2idx
    return word2idx

# Tokenizer Class
class Tokenizer:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_id = self.word2idx[self.pad_token]
        self.unk_id = self.word2idx[self.unk_token]
        self.sos_id = self.word2idx[self.sos_token]
        self.eos_id = self.word2idx[self.eos_token]
        self.vocab_size = len(word2idx)

    def encode(self, sentence, add_special_tokens=True):
        tokens = nltk.word_tokenize(sentence.lower())
        token_ids = [self.word2idx.get(token, self.unk_id) for token in tokens]
        if add_special_tokens:
            token_ids = [self.sos_id] + token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        for idx in token_ids:
            word = self.idx2word.get(idx, self.unk_token)
            if skip_special_tokens and word in {self.pad_token, self.sos_token, self.eos_token}:
                continue
            words.append(word)
        return " ".join(words)

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super(ImageEncoder, self).__init__()
        effnet = models.efficientnet_b4(weights='EfficientNet_B4_Weights.DEFAULT')
        self.backbone = effnet.features
        for param in self.backbone[:-2].parameters():
            param.requires_grad = False
        for param in self.backbone[-2:].parameters():
            param.requires_grad = True
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(2)
        self.transpose = lambda x: x.permute(0, 2, 1)
        self.project = nn.Linear(1792, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.transpose(x)
        x = self.project(x)
        return x

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Transformer Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None):
        _x = self.norm1(x + self.dropout(self.self_attn(x, x, x, attn_mask=tgt_mask)[0]))
        _x = self.norm2(_x + self.dropout(self.cross_attn(_x, enc_out, enc_out)[0]))
        out = self.norm3(_x + self.dropout(self.ff(_x)))
        return out

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.position_embedding(positions)
        tok_embed = self.token_embedding(x)
        return tok_embed + pos_embed

# Caption Decoder
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, num_heads, max_len, num_layers):
        super().__init__()
        self.pos_embed = PositionalEmbedding(vocab_size, max_len, embed_dim)
        self.dec_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def make_causal_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, tgt, enc_out):
        x = self.pos_embed(tgt)
        B, T, _ = x.shape
        mask = self.make_causal_mask(T).to(x.device)
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask=mask)
        logits = self.output_proj(x)
        return logits

# Full Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_encoder, transformer_encoder, decoder, tokenizer):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.transformer_encoder = transformer_encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def generate(self, image, max_length=100, beam_width=3, device='cuda', length_penalty=0.7):
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            img_features = self.cnn_encoder(image)
            encoded_img = self.transformer_encoder(img_features)
            beam = [([self.tokenizer.sos_id], 0.0)]
            for _ in range(max_length):
                candidates = []
                for seq, score in beam:
                    if seq[-1] == self.tokenizer.eos_id:
                        candidates.append((seq, score))
                        continue
                    input_ids = torch.tensor(seq).unsqueeze(0).to(device)
                    logits = self.decoder(input_ids, encoded_img)
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    topk_probs, topk_ids = probs.topk(beam_width)
                    for prob, idx in zip(topk_probs, topk_ids):
                        new_seq = seq + [idx.item()]
                        new_score = score + math.log(prob.item() + 1e-12)
                        candidates.append((new_seq, new_score))
                beam = sorted(candidates, key=lambda x: x[1] / ((len(x[0]) ** length_penalty) if length_penalty > 0 else 1), reverse=True)[:beam_width]
                if all(seq[-1] == self.tokenizer.eos_id for seq, _ in beam):
                    break
            best_seq = beam[0][0]
            return self.tokenizer.decode(best_seq, skip_special_tokens=True)

# Image preprocessing
img_size = (512, 512)
image_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Streamlit App
st.title("Medical Image Report Generator")
st.write("Upload a chest X-ray image to generate a medical report.")

# Option to enable GPU
use_gpu = st.checkbox("Use GPU (enable only if CUDA is properly configured)", value=False)
device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    st.info("Using GPU for inference. Ensure CUDA and drivers are compatible.")
else:
    st.info("Using CPU for inference to avoid CUDA issues.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# Load tokenizer vocabulary
EXPECTED_VOCAB_SIZE = 1810  # Based on model error (torch.Size([1810, 512]))
try:
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    word2idx = ensure_special_tokens(word2idx)
    vocab_size = len(word2idx)
    if vocab_size != EXPECTED_VOCAB_SIZE:
        st.error(f"Vocabulary size mismatch: expected {EXPECTED_VOCAB_SIZE} tokens, but got {vocab_size}. "
                 f"Please run generate_word2idx.py to create a compatible word2idx.pkl using indiana_reports.csv.")
        st.stop()
    st.success(f"Vocabulary loaded successfully with {vocab_size} tokens.")
except FileNotFoundError:
    st.error("Vocabulary file (word2idx.pkl) not found. Please run generate_word2idx.py to create it using indiana_reports.csv.")
    st.stop()
except Exception as e:
    st.error(f"Error loading word2idx.pkl: {e}")
    st.stop()

# Initialize tokenizer
try:
    tokenizer = Tokenizer(word2idx)
except Exception as e:
    st.error(f"Error initializing tokenizer: {e}")
    st.stop()

# Initialize model
embed_dim = 512
ff_dim = 512
num_heads = 8
num_decoder_layers = 4
max_len = 100

try:
    cnn_encoder = ImageEncoder(embed_dim=embed_dim).to(device)
    transformer_encoder = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim).to(device)
    decoder = CaptionDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        max_len=max_len,
        num_layers=num_decoder_layers
    ).to(device)
    model = ImageCaptioningModel(cnn_encoder, transformer_encoder, decoder, tokenizer).to(device)
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

# Load pre-trained model weights
try:
    model.load_state_dict(torch.load("medical_report_model.pt", map_location=device, weights_only=True))
    st.success("Pre-trained model loaded successfully.")
except FileNotFoundError:
    st.error("Model file (medical_report_model.pt) not found. Please ensure it is in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Process uploaded image and generate report
if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = image_transforms(image).to(device)
        
        # Generate report
        with st.spinner("Generating medical report..."):
            generated_report = model.generate(image_tensor, max_length=100, device=device)
        
        # Display image and report
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        st.subheader("Generated Medical Report")
        st.write(generated_report)
        
    except Exception as e:
        st.error(f"Error processing image or generating report: {e}")
else:
    st.info("Please upload an image to generate a report.")
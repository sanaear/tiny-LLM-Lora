import os
import torch
import torch.nn as nn
from torch.optim import Adam
from model.transformer import TinyTransformer

# =========================
# 1. Charger le corpus (chemin robuste)
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw", "corpus.txt")

if not os.path.exists(CORPUS_PATH):
    raise FileNotFoundError(f"Corpus introuvable : {CORPUS_PATH}")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("Corpus chargé. Taille :", len(text), "caractères")

# =========================
# 2. Tokenisation ultra simple (char-level)
# =========================

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

data = encode(text)

print("Vocab size :", vocab_size)

# =========================
# 3. Dataset minimal
# =========================

SEQ_LEN = 32
BATCH_SIZE = 16

def get_batch():
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix])
    return x, y

# =========================
# 4. Modèle
# =========================

model = TinyTransformer(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    max_len=SEQ_LEN
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Modèle prêt sur :", device)

# =========================
# 5. Entraînement
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4)

EPOCHS = 200

for step in range(EPOCHS):
    model.train()
    x, y = get_batch()
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:04d} | Loss: {loss.item():.4f}")

print("Entraînement terminé.")
# =========================
# 6. Génération de texte
# =========================

def generate(model, start, length=200):
    model.eval()
    ids = encode(start).unsqueeze(0).to(device)

    for _ in range(length):
        # garder seulement les derniers tokens
        ids_cond = ids[:, -SEQ_LEN:]

        with torch.no_grad():
            logits = model(ids_cond)
            next_token_logits = logits[:, -1, :]
            next_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        ids = torch.cat([ids, next_id], dim=1)

    return "".join([itos[i] for i in ids[0].tolist()])

# =========================
# 7. Test génération
# =========================

prompt = text[:20]  # début réel du corpus
generated = generate(model, prompt)

print("\n===== TEXTE GÉNÉRÉ =====\n")
print(generated)


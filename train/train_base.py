import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from model.transformer import TinyTransformer

# ==================================================
# 0. Reproductibilité
# ==================================================

torch.manual_seed(42)

# ==================================================
# 1. Chargement du corpus
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw", "corpus.txt")

if not os.path.exists(CORPUS_PATH):
    raise FileNotFoundError(f"Corpus introuvable : {CORPUS_PATH}")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Corpus chargé ({len(text)} caractères)")

# ==================================================
# 2. Tokenisation char-level (pédagogique)
# ==================================================

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids):
    return "".join([itos[i] for i in ids])

data = encode(text)

print(f"Vocab size : {vocab_size}")

# ==================================================
# 3. Split train / validation
# ==================================================

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# ==================================================
# 4. Dataset auto-régressif
# ==================================================

SEQ_LEN = 32
BATCH_SIZE = 16

def get_batch(split="train"):
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([source[i + 1:i + SEQ_LEN + 1] for i in ix])
    return x, y

# ==================================================
# 5. Modèle
# ==================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TinyTransformer(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    d_ff=512,
    max_len=SEQ_LEN,
    dropout=0.1
).to(device)

print(f"Modèle initialisé sur : {device}")

# ==================================================
# 6. Optimisation
# ==================================================

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4)

# ==================================================
# 7. Entraînement
# ==================================================

STEPS = 500

for step in range(STEPS):
    model.train()
    x, y = get_batch("train")
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        train_ppl = math.exp(loss.item())

        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch("val")
            logits_val = model(x_val.to(device))
            val_loss = criterion(
                logits_val.view(-1, vocab_size),
                y_val.to(device).view(-1)
            )
            val_ppl = math.exp(val_loss.item())

        print(
            f"Step {step:04d} | "
            f"Train PPL: {train_ppl:.2f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

print("Entraînement terminé.")

# ==================================================
# 8. Sauvegarde du modèle de base
# ==================================================

ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

ckpt_path = os.path.join(ckpt_dir, "tiny_transformer_base.pt")
torch.save(model.state_dict(), ckpt_path)

print(f"Modèle sauvegardé : {ckpt_path}")

# ==================================================
# 9. Génération de texte (greedy)
# ==================================================

def generate(model, start, length=200):
    model.eval()
    ids = encode(start).unsqueeze(0).to(device)

    for _ in range(length):
        ids_cond = ids[:, -SEQ_LEN:]

        with torch.no_grad():
            logits = model(ids_cond)
            next_logits = logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())

# ==================================================
# 10. Test génération
# ==================================================

prompt = text[:20]
generated = generate(model, prompt)

print("\n===== TEXTE GÉNÉRÉ =====\n")
print(generated)

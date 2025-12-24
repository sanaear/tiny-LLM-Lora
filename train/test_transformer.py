import torch
from model.transformer import TinyTransformer

vocab_size = 5000  # taille minimale pour tester
model = TinyTransformer(vocab_size)

x = torch.randint(0, vocab_size, (2, 10))  # batch=2, seq_len=10
out = model(x)
print("Sortie du modèle :", out.shape)

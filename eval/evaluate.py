import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== CONFIG =====
BASE_MODEL = "distilgpt2"
LORA_PATH = "../lora-out"   # dossier LoRA (même s'il est vide, ça marche sans)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ===== LOAD BASE MODEL =====
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.to(DEVICE)

# ===== LOAD LoRA (si disponible) =====
try:
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ Modèle LoRA chargé")
except:
    print("⚠️ Aucun LoRA trouvé, modèle de base utilisé")

model.eval()

# ===== PROMPTS DE TEST =====
prompts = [
    "What is a neural network?",
    "Explain artificial intelligence in simple terms.",
    "What is deep learning?"
]

# ===== GENERATION =====
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    print("\n" + "=" * 60)
    print("PROMPT:")
    print(prompt)
    print("\nRÉPONSE:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

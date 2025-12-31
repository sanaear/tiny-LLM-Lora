import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "distilgpt2"
LORA_PATH = "../lora-out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# QLoRA CONFIG (4-bit)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# =========================
# LOAD BASE MODEL (4-bit)
# =========================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)
base_model.eval()

# =========================
# LOAD LoRA
# =========================
try:
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print("✅ QLoRA chargé")
except:
    model = base_model
    print("⚠️ LoRA non trouvé → modèle quantifié seul")

model.eval()

# =========================
# PROMPTS
# =========================
prompts = [
    "What is a neural network?",
    "Explain artificial intelligence in simple terms.",
    "What is deep learning?"
]

# =========================
# GENERATION
# =========================
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\n" + "=" * 60)
    print("PROMPT:", prompt)
    print("RÉPONSE:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

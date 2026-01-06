import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ==================================================
# 0. Chemins & reproductibilité
# ==================================================

torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw", "corpus.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "lora-out")

# ==================================================
# 1. Chargement du corpus
# ==================================================

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

dataset = Dataset.from_dict({"text": [text]})

print(f"Corpus chargé ({len(text)} caractères)")

# ==================================================
# 2. Tokenizer (GPT-2)
# ==================================================

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_ds = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

# ==================================================
# 3. Configuration QLoRA (BitsAndBytes)
# ==================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ==================================================
# 4. Chargement du modèle pré-entraîné quantifié
# ==================================================

model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    quantization_config=bnb_config,
    device_map="auto"
)

# ==================================================
# 5. Configuration LoRA
# ==================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # projection QKV de GPT-2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================================================
# 6. Entraînement
# ==================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()

# ==================================================
# 7. Sauvegarde des poids LoRA
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ QLoRA sauvegardé dans {OUTPUT_DIR}")

# ==================================================
# 8. Génération de test
# ==================================================

prompt = "What is a neural network?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

print("\n===== TEXTE GÉNÉRÉ =====\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

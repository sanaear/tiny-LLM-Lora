import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# =========================
# 1. Charger le corpus
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw", "corpus.txt")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

dataset = Dataset.from_dict({"text": [text]})

# =========================
# 2. Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

# =========================
# 3. Modèle + LoRA
# =========================

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 4. Entraînement
# =========================

training_args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=1,
    num_train_epochs=30,
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

# ===== SAUVEGARDE LORA =====
output_dir = "../lora-out"

trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ LoRA sauvegardé dans : {output_dir}")


# =========================
# 5. Génération
# =========================

prompt = "what is a neural network"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7
)

print("\n===== TEXTE APRÈS LoRA =====\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


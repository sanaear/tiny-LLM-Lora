import streamlit as st
import torch
import math
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# =====================================================
# CONFIGURATION GLOBALE
# =====================================================

BASE_MODEL_NAME = "distilgpt2"
LORA_PATH = "./lora-out"
CORPUS_PATH = "./data/raw/corpus.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="TinyLLM — QLoRA Evaluation",
    layout="centered"
)

# =====================================================
# TITRE & CONTEXTE
# =====================================================

st.title("🧠 TinyLLM — Comparaison AVANT / APRÈS QLoRA")

st.markdown(
    """
    **Modèle de base** : DistilGPT-2 (quantifié 4-bit, NF4)  
    **Fine-tuning** : QLoRA (Low-Rank Adaptation)  
    **Objectif** : comparer génération et perplexité avant / après adaptation
    """
)

st.caption(
    "⚠️ La perplexité est calculée sur le corpus d'entraînement "
    "(comparaison relative, pas une évaluation générale)."
)

st.divider()

# =====================================================
# CHARGEMENT DU CORPUS
# =====================================================

@st.cache_data
def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

corpus_text = load_corpus(CORPUS_PATH)
st.success(f"📄 Corpus chargé ({len(corpus_text)} caractères)")

# =====================================================
# TOKENIZER
# =====================================================

@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = load_tokenizer()

# =====================================================
# MODELE DE BASE (QUANTIFIÉ 4-BIT)
# =====================================================

@st.cache_resource
def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()
    return model

base_model = load_base_model()

# =====================================================
# MODELE QLoRA
# =====================================================

@st.cache_resource
def load_lora_model():
    base = load_base_model()
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    return model

try:
    lora_model = load_lora_model()
    lora_loaded = True
    st.success("✅ Modèle QLoRA chargé avec succès")
except Exception:
    lora_loaded = False
    st.warning("⚠️ Modèle QLoRA non trouvé — seule la baseline est disponible")

st.info("🔧 Paramètres entraînés via LoRA : ~0.18 % du modèle")

# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def generate_text(model, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_perplexity(model, text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())

# =====================================================
# INTERFACE UTILISATEUR
# =====================================================

st.subheader("✏️ Paramètres de génération")

prompt = st.text_area(
    "Prompt",
    value="What is a neural network?",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    max_tokens = st.slider("Max new tokens", 20, 150, 80)

with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)

# =====================================================
# EXECUTION COMPARATIVE
# =====================================================

if st.button("🚀 Générer & Comparer"):

    st.divider()
    st.subheader("📌 AVANT QLoRA — Modèle de base")

    base_text = generate_text(
        base_model,
        prompt,
        max_tokens,
        temperature
    )

    base_ppl = compute_perplexity(base_model, corpus_text)

    st.write(base_text)
    st.metric("Perplexité (avant)", f"{base_ppl:.2f}")

    if lora_loaded:
        st.divider()
        st.subheader("📌 APRÈS QLoRA — Modèle adapté")

        lora_text = generate_text(
            lora_model,
            prompt,
            max_tokens,
            temperature
        )

        lora_ppl = compute_perplexity(lora_model, corpus_text)

        st.write(lora_text)
        st.metric("Perplexité (après)", f"{lora_ppl:.2f}")

        gain = base_ppl - lora_ppl
        st.success(f"📉 Gain de perplexité : {gain:.2f}")

    else:
        st.error("Comparaison impossible : QLoRA non chargé")

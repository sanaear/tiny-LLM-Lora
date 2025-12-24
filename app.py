import streamlit as st
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Tiny-LLM — Visualisation LoRA",
    layout="centered"
)

st.title("🤖 Tiny-LLM — Visualisation LoRA")
st.write("Comparaison de génération et de perplexité avant / après fine-tuning LoRA")

# =========================
# CHARGEMENT DES MODÈLES
# =========================
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    # Modèle de base PUR (sans LoRA)
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_model.eval()

    # Modèle LoRA (chargé séparément)
    lora_base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    lora_model = PeftModel.from_pretrained(
        lora_base,
        "lora-out"
    )
    lora_model.eval()

    return tokenizer, base_model, lora_model


tokenizer, base_model, lora_model = load_models()

# =========================
# FONCTION PERPLEXITÉ
# =========================
def compute_perplexity(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())

# =========================
# INTERFACE UTILISATEUR
# =========================
prompt = st.text_area(
    "Entrez un prompt :",
    "What is a neural network?",
    height=100
)

col_btn1, col_btn2 = st.columns(2)

# =========================
# GÉNÉRATION TEXTE
# =========================
if col_btn1.button("Générer"):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out_base = base_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        out_lora = lora_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟦 Avant LoRA")
        st.write(tokenizer.decode(out_base[0], skip_special_tokens=True))

    with col2:
        st.subheader("🟩 Après LoRA")
        st.write(tokenizer.decode(out_lora[0], skip_special_tokens=True))

# =========================
# PERPLEXITÉ
# =========================
if col_btn2.button("Afficher la perplexité"):
    ppl_base = compute_perplexity(base_model, tokenizer, prompt)
    ppl_lora = compute_perplexity(lora_model, tokenizer, prompt)

    st.markdown("### 📊 Perplexité")

    colp1, colp2, colp3 = st.columns(3)

    colp1.metric("Avant LoRA", f"{ppl_base:.2f}")
    colp2.metric("Après LoRA", f"{ppl_lora:.2f}")
    colp3.metric("Gain", f"{ppl_base - ppl_lora:.2f}")

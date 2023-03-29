import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

def paraphrase(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=256, num_return_sequences=1, no_repeat_ngram_size=3)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

st.title("Paraphrasing App")
input_text = st.text_area("Enter the text you want to paraphrase:", value="", height=None, max_chars=None, key=None)

if st.button("Paraphrase"):
    if input_text:
        paraphrased_text = paraphrase(input_text, model, tokenizer)
        st.subheader("Original Text:")
        st.write(input_text)
        st.subheader("Paraphrased Text:")
        st.write(paraphrased_text)
    else:
        st.warning("Please enter some text to paraphrase.")

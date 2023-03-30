import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

@st.cache(allow_output_mutation=True)
def load_model():
    device = "cpu"
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return model, tokenizer

model, tokenizer = load_model()

st.title("Paraphrasing App")
input_text = st.text_area("Enter the text you want to paraphrase:", value="", height=None, max_chars=None, key=None)

if st.button("Paraphrase"):
    if input_text:
        paraphrased_text = paraphrase(input_text)
        st.subheader("Original Text:")
        st.write(input_text)
        st.subheader("Paraphrased Text:")
        for i, p in enumerate(paraphrases):
            st.write("Option %s: %s" % (i+1, p))
    else:
        st.warning("Please enter some text to paraphrase.")


import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set Page Config
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")

# 1. Load the saved model and tokenizer
@st.cache_resource
def load_model():
    model_path = 'fake_review_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# 2. App Interface
st.title("🔍 Deceptive Hotel Review Detector")
st.markdown("Enter a hotel review below to check if it's likely Real or Fake.")

user_input = st.text_area("Review Text:", placeholder="Type or paste a review here...")

if st.button("Analyze Review"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # 3. Tokenization & Prediction
        inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True, max_length=256)
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        # 4. Display Result
        if prediction == 0:
            st.success("Real Review ✅")
            st.info("The model detected patterns consistent with genuine experiences.")
        else:
            st.error("Fake Review ❌")
            st.warning("The model detected exaggerated patterns or linguistic cues common in deceptive reviews.")

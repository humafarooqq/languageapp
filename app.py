# app.py
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the Hugging Face translation model and tokenizer
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-ur"  # Pretrained model for English to Urdu translation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Set up the Streamlit app title and description
st.title("English to Roman Urdu Translator")
st.markdown("<p style='color:blue; font-size:20px;'>Developed by Huma</p>", unsafe_allow_html=True)
st.markdown("<p style='color:red; font-size:15px;'>Powered by Hugging Face Transformer Model</p>", unsafe_allow_html=True)

# Input field for the user to enter English text
input_text = st.text_area("Enter English text to translate into Roman Urdu:", height=200)

# Button to trigger translation
if st.button("Translate"):
    if input_text:
        # Tokenize the input text
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        
        # Generate translation
        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        
        # Decode the generated translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the translated text
        st.subheader("Translation in Roman Urdu:")
        st.write(translation)
    else:
        st.error("Please enter some text to translate.")


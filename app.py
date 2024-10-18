import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model function
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-ur"  # Ensure the model is available on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Load tokenizer and model
tokenizer, model = load_model()

# Streamlit app layout
st.title("English to Roman Urdu Translator")
st.markdown("<p style='color:blue; font-size:20px;'>Developed by Huma</p>", unsafe_allow_html=True)
st.markdown("<p style='color:red; font-size:15px;'>Powered by Hugging Face Model</p>", unsafe_allow_html=True)

# Input for text
input_text = st.text_area("Enter English text to translate into Roman Urdu:", height=200)

# Button for translation
if st.button("Translate"):
    if input_text:
        # Tokenize input text
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        
        # Generate translation using model
        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display translation result
        st.subheader("Translation in Roman Urdu:")
        st.write(translation)
    else:
        st.error("Please enter some text to translate.")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained Hugging Face model and tokenizer
@st.cache_resource
def load_model():
    model_name = "your-huggingface-model-name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit App UI
st.title("English to Roman Urdu Translator")

# Input text
english_input = st.text_area("Enter English text:", height=150)

# Generate Roman Urdu translation
if st.button("Translate"):
    if english_input.strip() == "":
        st.error("Please enter some text to translate.")
    else:
        # Tokenize the input text
        inputs = tokenizer(english_input, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate output
        outputs = model.generate(**inputs, max_length=512)
        
        # Decode the generated text
        roman_urdu_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the result
        st.subheader("Roman Urdu Translation:")
        st.write(roman_urdu_translation)

# Footer
st.text("Powered by Hugging Face and Streamlit")

import streamlit as st
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Load the tokenizer and model
@st.cache_resource
def load_model():
    model_name = "abdulwaheed1/english-to-urdu-translation-mbart"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ur_PK")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Title and description of the app
st.title("English to Urdu Translator")
st.markdown("<p style='color:blue; font-size:20px;'>Developed by Huma</p>", unsafe_allow_html=True)
st.markdown("<p style='color:red; font-size:15px;'>Using fine-tuned MBart model for translation</p>", unsafe_allow_html=True)

# Input text area for user input
input_text = st.text_area("Enter English text to translate:", height=200)

# Button to trigger translation
if st.button("Translate"):
    if input_text:
        # Tokenize and translate the input text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=4, early_stopping=True)
        
        # Decode the translated text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the translated text
        st.subheader("Translation in Urdu:")
        st.write(translated_text)
    else:
        st.error("Please enter some text to translate.")

import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load Model and Preprocessing Objects
# -------------------------------

@st.cache_resource
def load_all():
    model = load_model("model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder


model, tokenizer, encoder = load_all()

# -------------------------------
# Text Cleaning Function
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# -------------------------------
# Prediction Function
# -------------------------------

def predict_sentiment(text):
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20)

    prediction = model.predict(padded)
    label_index = np.argmax(prediction)

    return encoder.inverse_transform([label_index])[0], prediction


# -------------------------------
# UI Design
# -------------------------------

st.set_page_config(page_title="WhatsApp Sentiment Analyzer", page_icon="💬")

st.title("💬 WhatsApp Sentiment Analyzer")
st.write("Analyze the sentiment of your messages using LSTM model")

# Input box
user_input = st.text_area("Enter your message here:")

# Button
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        result, prediction = predict_sentiment(user_input)

        # Display Result
        if result.lower() == "positive":
            st.success(f"😊 Sentiment: {result}")
        elif result.lower() == "negative":
            st.error(f"😡 Sentiment: {result}")
        else:
            st.info(f"😐 Sentiment: {result}")

        # Show confidence
        st.subheader("Prediction Confidence")
        st.write({
            "Negative": float(prediction[0][0]),
            "Neutral": float(prediction[0][1]),
            "Positive": float(prediction[0][2])
        })

# Footer
st.markdown("---")
st.caption("Built using LSTM, TensorFlow & Streamlit")
import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('svm_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 💅 Custom Animated Background CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;600&display=swap');

    body {
        font-family: 'Nunito', sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #B8EBD0, #FADADD, #D1CFE2, #F5E1FD);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        font-family: 'Nunito', sans-serif;
    }

    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .stTextArea textarea {
        background-color: #E0F7FA !important;  /* Powder blue */
        border-radius: 12px;
        font-size: 16px;
        padding: 10px;
        color: #333333;
    }

    .stButton>button {
        background-color: #6BCB77 !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        padding: 0.5em 1.5em;
        margin-top: 1em;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #4CAF50 !important;
        color: white !important;
    }

    .result-box {
        background-color: #ffffff;
        border-left: 8px solid #76C7C0;
        padding: 1em;
        border-radius: 10px;
        margin-top: 1em;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }

    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Sentiment Analyzer")
st.markdown("Write your product review below and let the AI predict its sentiment.")

# Text input
user_input = st.text_area("Enter your review here:", height=150)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf'[{string.punctuation}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Predict Button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = preprocess_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

        # Output based on binary class
        st.markdown("### Prediction Result")
        if prediction == 1:
            st.markdown('<div class="result-box"><h4>✅ Positive Sentiment</h4></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box"><h4>❌ Negative Sentiment</h4></div>', unsafe_allow_html=True)

        st.markdown("----")
        st.caption("🧠 Model used: Linear SVM | Vectorization: TF-IDF")
        st.caption("📌 Preprocessing: lowercasing, punctuation & stopword removal, stemming.")

# Footer
st.markdown("""
<hr style='margin-top: 40px;'>
<p style='text-align: center; font-size: 14px; color: gray;'>
Powered by Streamlit & Scikit-learn
</p>
""", unsafe_allow_html=True)
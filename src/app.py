import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
nltk.download('stopwords')
nltk.download('wordnet')

# Load model + vectorizer
model = joblib.load("src/fake_news_model.pkl")
vectorizer = joblib.load("src/tfidf_vectorizer.pkl")

# Text cleaning
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit page config
st.set_page_config(
    page_title="📰 Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

# Header
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E86C1;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #5D6D7E;
        margin-bottom: 40px;
    }
    .result-box {
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Detect whether a news headline or article is Real or Fake using Machine Learning</div>", unsafe_allow_html=True)

# Input box
user_input = st.text_area("📝 Enter News Headline or Article Text", height=200, placeholder="Type or paste news content here...")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Analyze"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            if prediction == 1:
                st.markdown("<div class='result-box' style='background-color:#D4EFDF; color:#1E8449;'>✅ Real News</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box' style='background-color:#F5B7B1; color:#922B21;'>❌ Fake News</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#808B96;'>Built with ❤️ using Streamlit, Python, and NLP</p>", unsafe_allow_html=True)

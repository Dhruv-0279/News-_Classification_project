import streamlit as st
import joblib
import spacy
import re

# --- 1. Page Configuration ---
st.set_page_config(page_title="News Classifier AI", page_icon="📰", layout="centered")

# --- 2. Load Models and NLP Tools ---
# Streamlit's @st.cache_resource ensures we only load these heavy files once
@st.cache_resource
def load_assets():
    model = joblib.load('models/logistic_regression_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    return model, tfidf, encoder, nlp

model, tfidf, encoder, nlp = load_assets()

# --- 3. Preprocessing Function ---
# (This must be exactly the same as the one in your notebook!)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    return " ".join(clean_tokens)

# --- 4. User Interface ---
st.title("📰 News Headline Classifier")
st.markdown("Enter a news headline below, and our Logistic Regression NLP model will instantly predict its category.")

# The input box
user_input = st.text_area("Headline Input:", placeholder="e.g., Apple announces new AI features for the upcoming iPhone...")

# The prediction button
if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter a headline first!")
    else:
        with st.spinner('Analyzing...'):
            # Step A: Clean the user's text
            cleaned_text = clean_text(user_input)
            
            # Step B: Vectorize the text using our saved TF-IDF
            vectorized_text = tfidf.transform([cleaned_text])
            
            # Step C: Get prediction and probabilities
            prediction = model.predict(vectorized_text)
            probabilities = model.predict_proba(vectorized_text)[0]
            
            # Step D: Translate the numerical prediction back to text
            predicted_category = encoder.inverse_transform(prediction)[0]
            confidence = max(probabilities) * 100
            
            # Display the result
            st.success(f"**Predicted Category:** {predicted_category}")
            st.info(f"**Model Confidence:** {confidence:.2f}%")
            
            # Optional: Show a warning if confidence is below our 70% business rule!
            if confidence < 70.0:
                st.warning("⚠️ Confidence is below 70%. In production, this would be routed to a human moderator.")
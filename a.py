import streamlit as st
import pandas as pd
from newspaper import Article
from transformers import pipeline

# Load pipelines (this may take a moment on the first run)
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource(show_spinner=False)
def load_sentiment():
    return pipeline("sentiment-analysis")

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment()

# Function to extract article text using newspaper3k
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error fetching article: {e}")
        return None

# Function to generate a summary from the article text
def generate_summary(text):
    # newspaper3k can sometimes extract long texts; limit input to summarizer
    max_input = 1024
    words = text.split()
    if len(words) > max_input:
        text = " ".join(words[:max_input])
    try:
        summary_list = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary_list[0]['summary_text']
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return "Summary generation failed."

# Function to analyze sentiment and compute an "importance score"
def analyze_sentiment(text):
    try:
        # Limit text length for performance
        sentiment = sentiment_analyzer(text[:512])[0]
        label = sentiment['label']
        score = sentiment['score']
        # Positive sentiment gives a positive score, negative sentiment gives a negative score.
        importance_score = score * 100 if label.upper() == "POSITIVE" else -score * 100
        return label, round(importance_score, 2)
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return "N/A", 0

# Function to run full analysis on a given URL
def analyze_article(url):
    article_text = extract_article_text(url)
    if not article_text or len(article_text) < 100:
        return None, None, None
    summary = generate_summary(article_text)
    sentiment_label, importance_score = analyze_sentiment(article_text)
    return summary, sentiment_label, importance_score

# Preloaded BBC article URLs (update with valid URLs as needed)
preloaded_articles = {
    "BBC Article 1": "https://www.bbc.com/news/world-us-canada-66801985",
    "BBC Article 2": "https://www.bbc.com/news/technology-66804779",
    "BBC Article 3": "https://www.bbc.com/news/science-environment-66799975"
}

# Streamlit App Layout
st.title("BBC News Article Analysis")
st.write("""
This app extracts BBC article text, generates a brief summary, and calculates an 
importance score based on the article’s sentiment (positive/negative) using text mining algorithms.
""")

# Sidebar: Choose between preloaded articles or a custom URL
analysis_mode = st.sidebar.radio("Select Analysis Mode", ["Preloaded Articles", "Custom URL"])

if analysis_mode == "Preloaded Articles":
    st.header("Preloaded BBC Articles Analysis")
    results = []
    for title, url in preloaded_articles.items():
        st.subheader(title)
        st.write(f"[Read the article]({url})")
        summary, sentiment_label, importance_score = analyze_article(url)
        if summary is None:
            st.error("Failed to extract article text or article is too short for analysis.")
            continue
        st.write("**Summary:**", summary)
        st.write("**Sentiment:**", sentiment_label)
        st.write("**Importance Score:**", importance_score)
        results.append({
            "Title": title,
            "URL": url,
            "Summary": summary,
            "Sentiment": sentiment_label,
            "Importance Score": importance_score
        })
        st.markdown("---")
    
    if results:
        st.subheader("Tabulated Results")
        df = pd.DataFrame(results)
        st.table(df)

elif analysis_mode == "Custom URL":
    st.header("Custom BBC Article Analysis")
    custom_url = st.text_input("Enter the URL of a BBC article:")
    if st.button("Generate Analysis") and custom_url:
        summary, sentiment_label, importance_score = analyze_article(custom_url)
        if summary is None:
            st.error("Failed to extract article text or the article text is too short for analysis.")
        else:
            st.write("**Summary:**", summary)
            st.write("**Sentiment:**", sentiment_label)
            st.write("**Importance Score:**", importance_score)
            st.success("Analysis complete!")

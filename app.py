import openai
import streamlit as st
import joblib

# OpenAI API Key
api_key = "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"

# Load Naive Bayes Model and Vectorizer
nb_model = joblib.load('multinomial_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon if you haven't already
nltk.download('vader_lexicon')

vader_lexicon_path = "vader_lexicon.txt"
def perform_sentiment_analysis(user_input):
    # Initialize the VADER sentiment analyzer with the custom lexicon file
    sid = SentimentIntensityAnalyzer(lexicon_file=vader_lexicon_path)
    
    # Get the polarity scores of the input text
    sentiment_scores = sid.polarity_scores(user_input)
    
    # Determine sentiment based on compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"



def ask_gpt3(question, conversation=[]):
    conversation.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        api_key=api_key
    )

    assistant_reply = response['choices'][0]['message']['content']
    return assistant_reply

def main():
    st.title("Sentiment Analysis & Chatbot App")
    st.sidebar.header("Chatbot")

    user_input = st.text_area("Enter a text for sentiment analysis:")
    
    if st.button("Analyze Sentiment"):
        sentiment_result = perform_sentiment_analysis(user_input)
        st.write(f"Sentiment: {sentiment_result}")

    user_question = st.text_input("Chatbot: Ask a question")
    if st.button("Ask GPT-3"):
        assistant_reply = ask_gpt3(user_question)
        st.write(f"Chatbot: {assistant_reply}")

if __name__ == "__main__":
    main()

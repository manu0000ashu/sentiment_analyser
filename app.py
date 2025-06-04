import streamlit as st
import nltk
import ssl
from textblob import TextBlob
import pyjokes
import random
import emoji
from nltk.tokenize import word_tokenize
import numpy as np

# Handle SSL certificate verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.warning("NLTK data download failed. Some features might be limited.")

# Initialize transformers with error handling
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
except Exception as e:
    st.error("Error loading AI models. Please try again later.")
    sentiment_analyzer = None
    emotion_classifier = None

class EmotionalSupportAssistant:
    def __init__(self):
        self.activities = [
            "Take a short walk outside",
            "Practice deep breathing for 5 minutes",
            "Listen to your favorite uplifting song",
            "Write down three things you're grateful for",
            "Do some light stretching exercises",
            "Call a friend or family member",
            "Draw or doodle something fun",
            "Read a few pages of an inspiring book",
            "Try a simple meditation exercise",
            "Make yourself a warm, comforting drink"
        ]
        
    def analyze_sentiment(self, text):
        try:
            if sentiment_analyzer:
                result = sentiment_analyzer(text)[0]
                return result
            else:
                # Fallback to TextBlob if transformer model fails
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                return {"label": "POSITIVE" if polarity > 0 else "NEGATIVE", "score": abs(polarity)}
        except Exception as e:
            return {"label": "NEUTRAL", "score": 0.5}
    
    def analyze_emotion(self, text):
        try:
            if emotion_classifier:
                emotions = emotion_classifier(text)[0]
                emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)
                return emotions_sorted[0]
            else:
                # Simple fallback emotion detection
                return {"label": "neutral", "score": 1.0}
        except Exception as e:
            return {"label": "neutral", "score": 1.0}
    
    def get_response(self, emotion):
        responses = {
            'joy': "I'm happy to see you're feeling good! Let's keep that positive energy going!",
            'sadness': "I understand you're feeling down. Remember that it's okay to feel this way, and I'm here to help.",
            'anger': "I can sense that you're frustrated. Let's try to work through this together.",
            'fear': "It's natural to feel afraid sometimes. Let's talk about what's worrying you.",
            'love': "That's wonderful! Love and connection are such important parts of life.",
            'surprise': "Life is full of surprises! Let's process this together.",
            'neutral': "I hear you. Would you like to explore your feelings a bit more?"
        }
        return responses.get(emotion['label'].lower(), "I'm here to listen and support you.")
    
    def suggest_activity(self):
        return random.choice(self.activities)
    
    def tell_joke(self):
        try:
            return pyjokes.get_joke()
        except:
            return "Why did the AI cross the road? To get to the other dataset! 😄"

def main():
    st.set_page_config(
        page_title="Emotional Support AI Assistant",
        page_icon="🤗",
        layout="wide"
    )
    
    st.title("Emotional Support AI Assistant 🤗")
    st.write("Share your thoughts and feelings with me, and I'll do my best to help you feel better!")
    
    assistant = EmotionalSupportAssistant()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("How are you feeling today?", height=100)
        if st.button("Send", key="send_button"):
            if user_input:
                # Analyze sentiment and emotion
                sentiment = assistant.analyze_sentiment(user_input)
                emotion = assistant.analyze_emotion(user_input)
                
                # Get assistant response
                response = assistant.get_response(emotion)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))
                
                # Display analysis
                st.write("---")
                st.subheader("Analysis")
                st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
                st.write(f"**Primary Emotion:** {emotion['label'].capitalize()} (Confidence: {emotion['score']:.2f})")
                
                # Provide support based on emotion
                if emotion['label'].lower() in ['sadness', 'anger', 'fear']:
                    st.write("---")
                    st.subheader("Let me help you feel better:")
                    st.info(assistant.suggest_activity())
                    st.success(f"Here's a joke to lighten the mood: {assistant.tell_joke()}")
    
    with col2:
        st.subheader("Chat History")
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")

if __name__ == "__main__":
    main()
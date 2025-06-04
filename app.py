import streamlit as st
import nltk
import ssl
from textblob import TextBlob
import pyjokes
import random
import emoji
from nltk.tokenize import word_tokenize
import numpy as np
import base64

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
except Exception as e:
    st.warning("NLTK data download failed. Some features might be limited.")

def add_bg_from_url():
    st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1505228395891-9a51e7e86bf6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3433&q=80");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        
        /* Modern Glass Effect Container */
        .glass-container {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* Custom styling for chat messages */
        .user-message {
            background: rgba(0, 149, 255, 0.1);
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            border-left: 4px solid #0095ff;
        }
        
        .assistant-message {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            border-left: 4px solid #00ff95;
        }
        
        /* Title styling */
        .main-title {
            color: #1E3D59;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Input box styling */
        .stTextArea textarea {
            border-radius: 15px;
            border: 2px solid rgba(28, 110, 140, 0.15);
            background: rgba(255, 255, 255, 0.9);
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #0095ff;
            box-shadow: 0 0 15px rgba(0, 149, 255, 0.2);
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 25px;
            padding: 10px 25px;
            background: linear-gradient(45deg, #0095ff, #00ff95);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 149, 255, 0.3);
        }
        
        /* Analysis section styling */
        .analysis-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Tip box styling */
        .tip-box {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #FFD93D;
        }
    </style>
    """, unsafe_allow_html=True)

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
        
        # Define emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'wonderful', 'great', 'awesome', 'fantastic'],
            'sadness': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'hurt', 'disappointed'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
            'love': ['love', 'loving', 'loved', 'care', 'caring', 'affection'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'astonished'],
        }
        
    def analyze_sentiment(self, text):
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.3:
                label = "POSITIVE"
            elif polarity < -0.3:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            return {"label": label, "score": abs(polarity)}
        except Exception as e:
            return {"label": "NEUTRAL", "score": 0.5}
    
    def analyze_emotion(self, text):
        try:
            text = text.lower()
            words = word_tokenize(text)
            
            # Count emotion keywords
            emotion_scores = {emotion: 0 for emotion in self.emotion_keywords}
            for word in words:
                for emotion, keywords in self.emotion_keywords.items():
                    if word in keywords:
                        emotion_scores[emotion] += 1
            
            # Get the dominant emotion
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            if max_emotion[1] == 0:
                return {"label": "neutral", "score": 1.0}
            return {"label": max_emotion[0], "score": 1.0}
        except Exception as e:
            return {"label": "neutral", "score": 1.0}
    
    def get_response(self, emotion):
        responses = {
            'joy': "I'm happy to see you're feeling good! Tell me more about what's making you happy!",
            'sadness': "I understand you're feeling down. Would you like to talk more about what's troubling you?",
            'anger': "I can sense that you're frustrated. What happened to make you feel this way?",
            'fear': "It's natural to feel afraid sometimes. Can you share what's causing your concern?",
            'love': "That's wonderful! I'd love to hear more about what's bringing love into your life!",
            'surprise': "Life is full of surprises! What's the unexpected thing that happened?",
            'neutral': "I'd love to hear more about that. What else is on your mind?"
        }
        return responses.get(emotion['label'].lower(), "Please tell me more about how you're feeling.")
    
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
    
    # Add background and custom styling
    add_bg_from_url()
    
    # Custom title with styling
    st.markdown('<h1 class="main-title">🌊 Emotional Support AI Assistant 🤗</h1>', unsafe_allow_html=True)
    
    # Wrap the introduction in a glass container
    st.markdown('<div class="glass-container">Share your thoughts and feelings with me, and I\'ll do my best to help you feel better!</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    assistant = EmotionalSupportAssistant()
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    # Display chat history first
    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("💭 Chat History")
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        # Text input area with a dynamic key
        user_input = st.text_area(
            "Type your message here...",
            key=f"user_input_{st.session_state.input_key}",
            height=100,
            help="Press Send or Ctrl+Enter to send your message"
        )
        
        # Send button
        if st.button("Send", key=f"send_button_{st.session_state.input_key}") or (user_input and st.session_state.get('enter_pressed', False)):
            if user_input.strip():
                # Analyze sentiment and emotion
                sentiment = assistant.analyze_sentiment(user_input)
                emotion = assistant.analyze_emotion(user_input)
                
                # Get assistant response
                response = assistant.get_response(emotion)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))
                
                # Display analysis in a styled container
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.subheader("📊 Analysis")
                st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
                st.write(f"**Primary Emotion:** {emotion['label'].capitalize()} (Confidence: {emotion['score']:.2f})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Provide support based on emotion
                if emotion['label'].lower() in ['sadness', 'anger', 'fear']:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.subheader("🌟 Let me help you feel better:")
                    st.info(assistant.suggest_activity())
                    st.success(f"Here's a joke to lighten the mood: {assistant.tell_joke()}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Increment the input key to clear the input field
                st.session_state.input_key += 1
                st.rerun()
        
        # Add a hint about continuing the conversation
        st.markdown("""
        <div class="tip-box">
        💡 <strong>Tip:</strong> After each response, you can:
        <ul>
            <li>Type a new message in the text box above</li>
            <li>Press 'Send' or Ctrl+Enter to continue the conversation</li>
            <li>Share more about your feelings or ask questions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
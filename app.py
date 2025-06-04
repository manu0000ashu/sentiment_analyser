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
            "Take a gentle walk in nature to clear your mind",
            "Listen to some soothing music that doesn't remind you of them",
            "Write down your feelings in a private journal",
            "Practice self-care with a warm bath or shower",
            "Try some deep breathing exercises for 5 minutes",
            "Call a close friend who's good at listening",
            "Make yourself your favorite comfort drink",
            "Watch a funny movie or show to lift your spirits",
            "Do some light exercise to boost endorphins",
            "Create art or express yourself creatively"
        ]
        
        # Enhanced emotion keywords with more nuanced emotions
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'wonderful', 'great', 'awesome', 'fantastic', 'blessed', 'grateful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'hurt', 'disappointed', 'heartbroken', 'lonely', 'breakup', 'broke up'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'upset', 'hate'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'uncertain', 'insecure'],
            'love': ['love', 'loving', 'loved', 'care', 'caring', 'affection', 'miss', 'relationship'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'astonished'],
            'confusion': ['confused', 'unsure', 'dont know', "don't know", 'lost', 'wondering'],
            'heartbreak': ['breakup', 'broke up', 'ex', 'relationship ended', 'heartbroken', 'dumped']
        }

        # Context-aware response templates
        self.response_templates = {
            'heartbreak': [
                "I'm so sorry you're going through a breakup. It's one of the most painful experiences, and your feelings are completely valid. Would you like to talk more about what you're feeling?",
                "Breakups can leave us feeling lost and hurt. Remember that healing takes time, and it's okay to not be okay right now. What's the hardest part for you?",
                "I hear how much pain you're in. Heartbreak is really tough, and you're brave for acknowledging these feelings. Would you like to share more about what happened?",
                "It's completely normal to feel down after a breakup. Your feelings matter, and this pain won't last forever. What do you need most right now - someone to listen, or maybe some suggestions for coping?"
            ],
            'sadness': [
                "I can hear the sadness in your words, and I want you to know that it's okay to feel this way. Would you like to tell me more about what's making you feel down?",
                "When we're feeling down, sometimes it helps to talk about it. I'm here to listen without judgment. What's weighing on your mind?",
                "I'm sorry you're feeling sad. Your feelings are valid, and you don't have to go through this alone. Would you like to explore what's causing these feelings?",
                "It takes courage to acknowledge when we're feeling down. I'm here to support you. What do you think triggered these feelings?"
            ],
            'confusion': [
                "It's okay to feel uncertain or lost sometimes. Would you like to try talking through what's on your mind? Sometimes that can help bring clarity.",
                "When we're not sure about our feelings, it can be overwhelming. Let's take it one step at time. What's the strongest emotion you're experiencing right now?",
                "Sometimes we need time to process our emotions, and that's perfectly normal. Would you like to explore these feelings together?",
                "It's natural to feel confused when processing difficult emotions. I'm here to listen and help you sort through your thoughts. What's the most pressing thing on your mind?"
            ]
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
    
    def get_response(self, emotion_analysis):
        # Get the detected emotion
        emotion = emotion_analysis['label'].lower()
        
        # Check for specific keywords in the last message
        last_message = st.session_state.chat_history[-1][1] if st.session_state.chat_history else ""
        last_message = last_message.lower()
        
        # Check for context-specific situations
        if any(word in last_message for word in ['breakup', 'broke up', 'ex', 'relationship ended']):
            return random.choice(self.response_templates['heartbreak'])
        elif emotion == 'sadness' or 'sad' in last_message or 'down' in last_message:
            return random.choice(self.response_templates['sadness'])
        elif 'dont know' in last_message.replace("'", "") or 'confused' in last_message or 'unsure' in last_message:
            return random.choice(self.response_templates['confusion'])
        
        # Default responses based on emotion
        responses = {
            'joy': "I'm glad you're feeling positive! What's bringing this happiness into your life?",
            'sadness': "I hear the sadness in your words. Would you like to talk about what's troubling you?",
            'anger': "I can sense your frustration. What happened to make you feel this way?",
            'fear': "It's okay to feel anxious or worried. Would you like to share what's causing these feelings?",
            'love': "Love is such a powerful emotion. Would you like to tell me more about these feelings?",
            'surprise': "Unexpected things can really impact us. How are you processing this surprise?",
            'neutral': "I'm here to listen. What's on your mind right now?"
        }
        return responses.get(emotion, "I'm here to listen and support you. Would you like to tell me more?")
    
    def suggest_activity(self):
        return random.choice(self.activities)
    
    def tell_joke(self):
        try:
            # Get a joke that's appropriate for someone feeling down
            jokes = [
                "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a bear with no teeth? A gummy bear!",
                "Why did the cookie go to the doctor? Because it was feeling crumbly!",
                "What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "What do you call a snowman with a six-pack? An abdominal snowman!",
                "Why did the math book look so sad? Because it had too many problems!"
            ]
            return random.choice(jokes)
        except:
            return "Why did the cookie go to the doctor? Because it was feeling crumbly! 🍪"

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
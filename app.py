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
from nltk.util import ngrams
from nltk.probability import FreqDist
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    nltk.download('stopwords', quiet=True)
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

class EmotionalResponseGenerator:
    def __init__(self):
        # Emotional response dataset
        self.emotion_responses = {
            "sadness": [
                {"text": "I understand you're going through a difficult time. Your feelings of sadness are valid.", "context": "general"},
                {"text": "It's okay to feel sad. Would you like to talk about what's troubling you?", "context": "general"},
                {"text": "I hear the pain in your words. Remember that it's okay to take time to process these feelings.", "context": "general"},
                {"text": "Sadness can feel overwhelming, but you don't have to face it alone. I'm here to listen.", "context": "general"},
                {"text": "Sometimes sadness needs to be felt fully before we can begin to heal. What do you need right now?", "context": "general"}
            ],
            "heartbreak": [
                {"text": "Breakups can be incredibly painful. Your heart needs time to heal, and that's perfectly normal.", "context": "breakup"},
                {"text": "I hear how much this breakup is affecting you. It's okay to grieve the relationship.", "context": "breakup"},
                {"text": "The end of a relationship can feel like losing a part of yourself. Give yourself permission to feel these emotions.", "context": "breakup"},
                {"text": "Heartbreak is one of the deepest pains we can experience. Your feelings are completely valid.", "context": "breakup"},
                {"text": "It's natural to feel lost after a breakup. Would you like to talk about what you're experiencing?", "context": "breakup"}
            ],
            "anger": [
                {"text": "I can sense your frustration. It's okay to feel angry about this situation.", "context": "general"},
                {"text": "Your anger is valid. Would you like to explore what's triggering these feelings?", "context": "general"},
                {"text": "Sometimes anger can be a sign that our boundaries have been crossed. What happened?", "context": "general"},
                {"text": "It's natural to feel angry when we're hurt. I'm here to listen without judgment.", "context": "general"}
            ],
            "anxiety": [
                {"text": "Anxiety can feel overwhelming. Let's take it one step at a time.", "context": "general"},
                {"text": "When anxiety hits, it's important to remember that you're not alone in this.", "context": "general"},
                {"text": "I hear that you're feeling anxious. Would you like to talk about what's causing these feelings?", "context": "general"},
                {"text": "Anxiety is a natural response to stress. What helps you feel grounded when these feelings arise?", "context": "general"}
            ],
            "confusion": [
                {"text": "It's okay to feel uncertain. Sometimes talking things through can help bring clarity.", "context": "general"},
                {"text": "When we're confused, it can help to break things down into smaller pieces. What's the main thing troubling you?", "context": "general"},
                {"text": "Feeling lost is a natural part of processing complex emotions. Would you like to explore these feelings together?", "context": "general"}
            ],
            "hope": [
                {"text": "I'm glad you're feeling hopeful. What's giving you this positive outlook?", "context": "general"},
                {"text": "Hope is a powerful emotion that can help us through difficult times. Tell me more about what's inspiring you.", "context": "general"},
                {"text": "It's wonderful to hear that spark of hope in your words. What's changed?", "context": "general"}
            ]
        }

        # Initialize n-gram models
        self.ngram_models = defaultdict(FreqDist)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self._train_ngram_models()

    def _train_ngram_models(self):
        # Train n-gram models for each emotion
        for emotion, responses in self.emotion_responses.items():
            texts = [response["text"] for response in responses]
            for text in texts:
                tokens = word_tokenize(text.lower())
                for n in range(1, 4):  # Use 1 to 3-grams
                    text_ngrams = list(ngrams(tokens, n))
                    self.ngram_models[emotion].update(text_ngrams)

    def analyze_emotion_ngrams(self, text):
        tokens = word_tokenize(text.lower())
        text_ngrams = []
        for n in range(1, 4):
            text_ngrams.extend(list(ngrams(tokens, n)))

        # Calculate emotion scores based on n-gram overlap
        emotion_scores = defaultdict(float)
        for emotion, freq_dist in self.ngram_models.items():
            for ngram in text_ngrams:
                emotion_scores[emotion] += freq_dist[ngram]

        # Normalize scores
        total = sum(emotion_scores.values()) or 1
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # Get the dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return {
            "label": dominant_emotion[0],
            "score": dominant_emotion[1],
            "all_scores": emotion_scores
        }

    def get_best_response(self, text, emotion_analysis):
        # Get relevant responses for the detected emotion
        emotion = emotion_analysis["label"]
        relevant_responses = self.emotion_responses.get(emotion, [])
        
        if not relevant_responses:
            return "I'm here to listen and support you. Would you like to tell me more?"

        # Create TF-IDF vectors
        all_responses = [resp["text"] for resp in relevant_responses]
        all_responses.append(text)
        tfidf_matrix = self.vectorizer.fit_transform(all_responses)
        
        # Calculate similarity between input and responses
        user_vector = tfidf_matrix[-1]
        response_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(user_vector, response_vectors)
        
        # Get the most similar response
        best_response_idx = similarities.argmax()
        return relevant_responses[best_response_idx]["text"]

class EmotionalSupportAssistant:
    def __init__(self):
        self.response_generator = EmotionalResponseGenerator()
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
        return self.response_generator.analyze_emotion_ngrams(text)

    def get_response(self, text, emotion_analysis):
        return self.response_generator.get_best_response(text, emotion_analysis)

    def suggest_activity(self):
        return random.choice(self.activities)

    def tell_joke(self):
        try:
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
                
                # Get assistant response using the enhanced response generator
                response = assistant.get_response(user_input, emotion)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))
                
                # Display analysis in a styled container
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.subheader("📊 Analysis")
                st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
                st.write(f"**Primary Emotion:** {emotion['label'].capitalize()} (Confidence: {emotion['score']:.2f})")
                
                # Display emotion distribution
                if 'all_scores' in emotion:
                    st.write("**Emotion Distribution:**")
                    for emo, score in emotion['all_scores'].items():
                        st.write(f"- {emo.capitalize()}: {score:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Provide support based on emotion
                if emotion['label'].lower() in ['sadness', 'heartbreak', 'anger', 'anxiety']:
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
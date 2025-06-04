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
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import spacy
from rasa.nlu.model import Interpreter
import os

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
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.warning("NLTK data download failed. Some features might be limited.")

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

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

class AdvancedNLPProcessor:
    def __init__(self):
        # Initialize transformers
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Intent recognition patterns
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon'],
            'farewell': ['bye', 'goodbye', 'see you', 'talk to you later', 'have to go'],
            'gratitude': ['thank you', 'thanks', 'appreciate it', 'grateful'],
            'help_request': ['help', 'need advice', 'what should i do', 'can you help'],
            'sharing_emotion': ['feel', 'feeling', 'felt', 'am sad', 'am happy', 'am angry'],
            'question': ['why', 'what', 'how', 'when', 'where', 'who']
        }

    def analyze_intent(self, text):
        # Use SpaCy for advanced text processing
        doc = nlp(text.lower())
        
        # Extract key phrases and dependencies
        key_phrases = []
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text)
        
        # Detect intent using both pattern matching and dependency parsing
        intent_scores = defaultdict(float)
        
        # Pattern-based scoring
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if pattern in text.lower():
                    intent_scores[intent] += 1.0
        
        # Dependency-based scoring
        for token in doc:
            if token.dep_ in ['ROOT', 'dobj', 'nsubj']:
                for intent, patterns in self.intents.items():
                    if token.text.lower() in patterns:
                        intent_scores[intent] += 0.5
        
        # Get the dominant intent
        if intent_scores:
            dominant_intent = max(intent_scores.items(), key=lambda x: x[1])
            return {
                'intent': dominant_intent[0],
                'confidence': dominant_intent[1],
                'all_intents': dict(intent_scores),
                'key_phrases': key_phrases
            }
        return {'intent': 'general', 'confidence': 1.0, 'key_phrases': key_phrases}

    def analyze_sentiment(self, text):
        # Combine multiple sentiment analysis approaches
        try:
            # Transformer-based sentiment
            transformer_sentiment = self.sentiment_analyzer(text)[0]
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine scores
            if transformer_sentiment['label'] == 'POSITIVE':
                combined_score = (transformer_sentiment['score'] + max(textblob_polarity, 0)) / 2
                label = "POSITIVE"
            else:
                combined_score = (transformer_sentiment['score'] + max(-textblob_polarity, 0)) / 2
                label = "NEGATIVE"
                
            return {
                'label': label,
                'score': combined_score,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            return {"label": "NEUTRAL", "score": 0.5, "subjectivity": 0.5}

    def analyze_emotion(self, text):
        try:
            # Get emotion classification from transformer
            emotion_result = self.emotion_classifier(text)[0]
            
            return {
                'label': emotion_result['label'],
                'score': emotion_result['score'],
                'detailed_analysis': self.get_emotional_aspects(text)
            }
        except Exception as e:
            return {"label": "neutral", "score": 1.0}

    def get_emotional_aspects(self, text):
        doc = nlp(text)
        aspects = {
            'intensity': self._get_intensity(doc),
            'key_emotions': self._extract_emotion_words(doc),
            'temporal_context': self._get_temporal_context(doc),
            'certainty': self._assess_certainty(doc)
        }
        return aspects

    def _get_intensity(self, doc):
        intensity_markers = ['very', 'really', 'extremely', 'so', 'totally', 'absolutely']
        return sum(1 for token in doc if token.text.lower() in intensity_markers)

    def _extract_emotion_words(self, doc):
        emotion_words = []
        for token in doc:
            if token.pos_ in ['ADJ', 'VERB'] and token.text.lower() in self.emotion_classifier.model.config.id2label.values():
                emotion_words.append(token.text)
        return emotion_words

    def _get_temporal_context(self, doc):
        temporal_markers = {
            'past': ['was', 'were', 'had', 'felt', 'did'],
            'present': ['am', 'is', 'are', 'feel', 'do'],
            'future': ['will', 'going to', 'planning', 'hope']
        }
        
        context = defaultdict(int)
        for token in doc:
            for timeframe, markers in temporal_markers.items():
                if token.text.lower() in markers:
                    context[timeframe] += 1
        return dict(context)

    def _assess_certainty(self, doc):
        certainty_markers = {
            'high': ['definitely', 'certainly', 'absolutely', 'sure'],
            'low': ['maybe', 'perhaps', 'might', 'possibly', 'guess']
        }
        
        certainty = defaultdict(int)
        for token in doc:
            for level, markers in certainty_markers.items():
                if token.text.lower() in markers:
                    certainty[level] += 1
        return dict(certainty)

class EmotionalResponseGenerator:
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load response templates
        self.response_templates = {
            "sadness": [
                {"text": "I understand you're going through a difficult time. Your feelings of sadness are valid.", "context": "general"},
                {"text": "It's okay to feel sad. Would you like to talk about what's troubling you?", "context": "general"},
                {"text": "I hear the pain in your words. Remember that it's okay to take time to process these feelings.", "context": "general"},
                {"text": "Sadness can feel overwhelming, but you don't have to face it alone. I'm here to listen.", "context": "general"}
            ],
            "heartbreak": [
                {"text": "Breakups can be incredibly painful. Your heart needs time to heal, and that's perfectly normal.", "context": "breakup"},
                {"text": "I hear how much this breakup is affecting you. It's okay to grieve the relationship.", "context": "breakup"},
                {"text": "The end of a relationship can feel like losing a part of yourself. Give yourself permission to feel these emotions.", "context": "breakup"}
            ],
            "anger": [
                {"text": "I can sense your frustration. It's okay to feel angry about this situation.", "context": "general"},
                {"text": "Your anger is valid. Would you like to explore what's triggering these feelings?", "context": "general"},
                {"text": "Sometimes anger can be a sign that our boundaries have been crossed. What happened?", "context": "general"}
            ],
            "anxiety": [
                {"text": "Anxiety can feel overwhelming. Let's take it one step at a time.", "context": "general"},
                {"text": "When anxiety hits, it's important to remember that you're not alone in this.", "context": "general"},
                {"text": "I hear that you're feeling anxious. Would you like to talk about what's causing these feelings?", "context": "general"}
            ],
            "joy": [
                {"text": "Your happiness is contagious! What's bringing you joy right now?", "context": "general"},
                {"text": "It's wonderful to hear you're feeling good! Would you like to share more about what's making you happy?", "context": "general"},
                {"text": "I'm glad you're experiencing such positive emotions! What's contributed to this happiness?", "context": "general"}
            ]
        }
        
        # Create embeddings for all responses
        self.response_embeddings = {}
        for emotion, responses in self.response_templates.items():
            self.response_embeddings[emotion] = {
                'texts': [r['text'] for r in responses],
                'embeddings': self.sentence_model.encode([r['text'] for r in responses])
            }

    def get_best_response(self, text, analysis):
        # Get text embedding
        text_embedding = self.sentence_model.encode([text])[0]
        
        # Get emotion and intent
        emotion = analysis['emotion']['label']
        intent = analysis['intent']['intent']
        
        # Select relevant responses based on emotion and intent
        relevant_responses = self.response_templates.get(emotion, self.response_templates['sadness'])
        
        if not relevant_responses:
            return "I'm here to listen and support you. Would you like to tell me more?"
        
        # Get embeddings for relevant responses
        response_embeddings = self.response_embeddings[emotion]['embeddings']
        response_texts = self.response_embeddings[emotion]['texts']
        
        # Calculate similarities
        similarities = cosine_similarity([text_embedding], response_embeddings)[0]
        
        # Get the most similar response
        best_response_idx = similarities.argmax()
        
        return response_texts[best_response_idx]

class EmotionalSupportAssistant:
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
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

    def analyze_input(self, text):
        # Comprehensive analysis of user input
        return {
            'sentiment': self.nlp_processor.analyze_sentiment(text),
            'emotion': self.nlp_processor.analyze_emotion(text),
            'intent': self.nlp_processor.analyze_intent(text)
        }

    def get_response(self, text, analysis):
        return self.response_generator.get_best_response(text, analysis)

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
                # Comprehensive analysis of user input
                analysis = assistant.analyze_input(user_input)
                
                # Get assistant response using the enhanced response generator
                response = assistant.get_response(user_input, analysis)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))
                
                # Display analysis in a styled container
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.subheader("📊 Analysis")
                
                # Display sentiment analysis
                sentiment = analysis['sentiment']
                st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
                st.write(f"**Subjectivity:** {sentiment['subjectivity']:.2f}")
                
                # Display emotion analysis
                emotion = analysis['emotion']
                st.write(f"**Primary Emotion:** {emotion['label'].capitalize()} (Confidence: {emotion['score']:.2f})")
                
                # Display intent analysis
                intent = analysis['intent']
                st.write(f"**Detected Intent:** {intent['intent'].capitalize()} (Confidence: {intent['confidence']:.2f})")
                
                # Display detailed emotional aspects
                if 'detailed_analysis' in emotion:
                    st.write("**Emotional Analysis Details:**")
                    details = emotion['detailed_analysis']
                    st.write(f"- Emotional Intensity: {details['intensity']}")
                    st.write(f"- Key Emotion Words: {', '.join(details['key_emotions']) if details['key_emotions'] else 'None detected'}")
                    st.write(f"- Temporal Context: {details['temporal_context']}")
                    st.write(f"- Certainty Levels: {details['certainty']}")
                
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
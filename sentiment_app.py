import streamlit as st
import torch
import os
import time
from transformers import BertTokenizer

# Import local modules
from load_sentiment_model import create_model_from_checkpoint, predict_sentiment

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define function to load the model
@st.cache_resource
def load_model(model_path, config_path):
    """Load model with caching to avoid reloading on each interaction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        return None, None
    
    if not os.path.exists(config_path):
        st.error(f"Error: Config file not found at {config_path}")
        return None, None
    
    # Create model and load weights with correct architecture detection
    try:
        model, tokenizer = create_model_from_checkpoint(config_path, model_path, device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Set up sidebar
def sidebar_settings():
    st.sidebar.title("Sentiment Analysis Settings")
    
    # Model path input with default value
    model_path = st.sidebar.text_input(
        "Model Path",
        value="runs/custom_model/kaggle2.pt",
        help="Path to the model weights file"
    )
    
    # Config path input with default value
    config_path = st.sidebar.text_input(
        "Config Path",
        value="runs/custom_model/config.json",
        help="Path to the model configuration file"
    )
    
    # Max text length
    max_length = st.sidebar.slider(
        "Max Token Length",
        min_value=64,
        max_value=512,
        value=128,
        step=32,
        help="Maximum number of tokens to process"
    )
    
    # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses a Multi-Headed Attention based sentiment analysis model "
        "with LSTM layers to classify text as Negative, Neutral, or Positive."
    )
    
    # Display device information
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.markdown(f"**Running on:** {device}")
    
    return model_path, config_path, max_length

# Main app function
def main():
    # Title and intro
    st.title("📊 Sentiment Analysis Tool")
    st.markdown(
        "Analyze the sentiment of text using an enhanced deep learning model. "
        "Enter a single sentence or multiple sentences to analyze."
    )
    
    # Get settings from sidebar
    model_path, config_path, max_length = sidebar_settings()
    
    # Load the model
    with st.spinner("Loading model... This may take a moment."):
        model, tokenizer, device = load_model(model_path, config_path)
    
    if model is None or tokenizer is None:
        st.warning("Please check model and config paths in the sidebar.")
        return
    
    # Success message when model is loaded
    st.success("Model loaded successfully!")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Single Sentence", "Multiple Sentences"],
        horizontal=True
    )
    
    # Process input based on selected method
    if input_method == "Single Sentence":
        process_single_input(model, tokenizer, device, max_length)
    else:
        process_multiple_inputs(model, tokenizer, device, max_length)

def process_single_input(model, tokenizer, device, max_length):
    # Input for a single sentence
    text = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="Type your sentence here... (e.g., 'I really enjoyed this movie!')"
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", key="analyze_single"):
        if text.strip():
            with st.spinner("Analyzing sentiment..."):
                # Small delay to show the spinner
                time.sleep(0.5)
                result = predict_sentiment(text, model, tokenizer, device, max_length)
            
            # Display results
            display_sentiment_result(result)
        else:
            st.warning("Please enter some text to analyze.")

def process_multiple_inputs(model, tokenizer, device, max_length):
    # Input for multiple sentences
    text_input = st.text_area(
        "Enter multiple sentences (one per line):",
        height=150,
        placeholder="Type your sentences here, one per line:\nThe service was excellent.\nI had a terrible experience.\nThe product quality was acceptable."
    )
    
    if text_input.strip():
        # Split by newlines and filter out empty lines
        sentences = [s.strip() for s in text_input.split('\n') if s.strip()]
        st.write(f"**{len(sentences)} sentences detected**")
        
        # Analyze all sentences and store results
        all_results = []
        if st.button("Analyze All Sentences", key="analyze_multiple"):
            progress_bar = st.progress(0)
            
            for i, sentence in enumerate(sentences):
                # Update progress bar
                progress = (i + 1) / len(sentences)
                progress_bar.progress(progress)
                
                # Analyze sentiment
                result = predict_sentiment(sentence, model, tokenizer, device, max_length)
                all_results.append((sentence, result))
                
                # Small delay to show progress
                time.sleep(0.1)
            
            # Clear progress bar when done
            progress_bar.empty()
            
            # Store results in session state
            st.session_state['results'] = all_results
            st.session_state['current_index'] = 0
        
        # Display results one by one
        if 'results' in st.session_state and st.session_state['results']:
            st.subheader("Analysis Results")
            
            results = st.session_state['results']
            current_index = st.session_state['current_index']
            
            # Show current sentence and its sentiment
            st.markdown(f"**Sentence {current_index + 1}/{len(results)}:**")
            st.info(results[current_index][0])
            
            # Display sentiment for current sentence
            display_sentiment_result(results[current_index][1])
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if current_index > 0:
                    if st.button("← Previous"):
                        st.session_state['current_index'] -= 1
                        st.rerun()
            
            with col2:
                if current_index < len(results) - 1:
                    if st.button("Next →"):
                        st.session_state['current_index'] += 1
                        st.rerun()

def display_sentiment_result(result):
    """Display sentiment analysis results with visual elements"""
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    # Set color based on sentiment
    color = {
        'Negative': 'red',
        'Neutral': 'blue',
        'Positive': 'green'
    }.get(sentiment, 'gray')
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display sentiment emoji and label
        emoji_map = {
            'Negative': '😔',
            'Neutral': '😐',
            'Positive': '😊'
        }
        emoji = emoji_map.get(sentiment, '❓')
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{sentiment}</h3>", unsafe_allow_html=True)
    
    with col2:
        # Display confidence percentage
        st.markdown("**Confidence:**")
        st.progress(confidence / 100)
        st.text(f"{confidence:.2f}%")
        
        # Display all probabilities
        st.markdown("**Probability Distribution:**")
        
        # Create a horizontal bar chart for probabilities
        probs = result['probabilities']
        for sent, prob in probs.items():
            sent_color = {
                'Negative': 'red',
                'Neutral': 'blue',
                'Positive': 'green'
            }.get(sent, 'gray')
            
            st.markdown(f"{sent}: {prob:.2f}%")
            st.progress(prob / 100)

if __name__ == "__main__":
    main()

import torch
import argparse
import json
import os
import numpy as np
from utils import load_model
from models import SentimentModel
from enhanced_models import EnhancedSentimentModel, BalancedSentimentModel
from dataset import prepare_tokenizer

class SentimentPredictor:
    """A unified interface for sentiment prediction that can use any model type"""
    
    def __init__(self, model_path=None, model_type=None):
        """
        Initialize the predictor with a specific model
        
        Args:
            model_path: Path to the model checkpoint file
            model_type: Type of model ('original', 'enhanced', or 'balanced')
                        If None, will try to auto-detect from model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load dataset for tokenizer initialization
        data_path = "data/final_dataset.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        self.tokenizer, _ = prepare_tokenizer(data)
        
        # Determine model path if not provided
        if model_path is None:
            model_path = self._find_best_model(model_type)
            if model_path is None:
                raise ValueError("No suitable model checkpoint found.")
        
        # Load the model
        self._load_model(model_path, model_type)
        
    def _find_best_model(self, model_type):
        """Find the best available model based on type"""
        print(f"Looking for best {model_type if model_type else 'any'} model...")
        
        # Map model types to run directories
        run_dirs = sorted([os.path.join("runs", d) for d in os.listdir("runs")], key=os.path.getmtime)
        
        if model_type == 'original':
            # Find original model runs (without 'enhanced' in directory name)
            candidate_runs = [d for d in run_dirs if 'enhanced' not in d.lower()]
        elif model_type == 'enhanced' or model_type == 'balanced':
            # Find enhanced model runs (with 'enhanced' in directory name)
            candidate_runs = [d for d in run_dirs if 'enhanced' in d.lower()]
        else:
            # Consider all runs
            candidate_runs = run_dirs
        
        for run_dir in reversed(candidate_runs):  # Start with most recent
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                continue
                
            # Look for best model checkpoints
            best_models = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model')]
            if best_models:
                best_model = sorted(best_models, 
                                  key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), 
                                  reverse=True)[0]
                return os.path.join(checkpoint_dir, best_model)
        
        return None
        
    def _load_model(self, model_path, model_type=None):
        """Load model from checkpoint"""
        print(f"Loading model from {model_path}")
        
        # Determine model class based on type
        if model_type == 'original':
            model_class = SentimentModel
        elif model_type == 'enhanced':
            model_class = EnhancedSentimentModel
        elif model_type == 'balanced':
            model_class = BalancedSentimentModel
        else:
            # Try to auto-detect model type
            model_classes = [BalancedSentimentModel, EnhancedSentimentModel, SentimentModel]
            
            for cls in model_classes:
                try:
                    self.model, _ = load_model(model_path, cls, self.tokenizer, self.device)
                    print(f"Successfully loaded model as {cls.__name__}")
                    return
                except Exception as e:
                    print(f"Failed to load as {cls.__name__}: {e}")
            
            raise ValueError("Could not load model with any of the available model classes")
            
        # Load with specified model class
        self.model, _ = load_model(model_path, model_class, self.tokenizer, self.device)
        print(f"Successfully loaded {model_class.__name__}")
    
    def predict(self, text):
        """
        Predict sentiment for a given text
        
        Args:
            text: The text to analyze
            
        Returns:
            A tuple of (sentiment_label, confidence_scores)
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # Map to sentiment labels
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        predicted_sentiment = sentiment_labels[predicted_class.item()]
        
        # Return prediction and confidence scores
        return {
            'sentiment': predicted_sentiment,
            'confidence': confidence.item(),
            'probabilities': {
                label: prob.item() for label, prob in zip(sentiment_labels, probabilities[0])
            }
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis Prediction')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--model', type=str, choices=['original', 'enhanced', 'balanced'], 
                        default=None, help='Model type to use')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SentimentPredictor(args.model_path, args.model)
    
    # Get text from argument or prompt user
    text = args.text if args.text else input("Enter text for sentiment analysis: ")
    
    # Make prediction
    result = predictor.predict(text)
    
    # Display results
    print("\nSentiment Analysis Results:")
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nProbabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()

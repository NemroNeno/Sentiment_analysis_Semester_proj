import torch
import argparse
import os
import json
from transformers import BertTokenizer
from models import SentimentModel
from enhanced_models import EnhancedSentimentModel, BalancedSentimentModel
from utils import load_model
from dataset import load_data, prepare_tokenizer

def predict_sentiment(text, model, tokenizer, device, max_length=512):
    """Make a sentiment prediction for a given text"""
    # Tokenize input text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
    
    # Map prediction to sentiment label
    sentiment_map = {
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    }
    
    sentiment = sentiment_map.get(prediction.item(), 'Unknown')
    
    # Get confidence scores (softmax probabilities)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidence = probabilities[prediction.item()].item() * 100
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'Negative': probabilities[0].item() * 100,
            'Neutral': probabilities[1].item() * 100,
            'Positive': probabilities[2].item() * 100
        }
    }

class SentimentAnalyzer:
    """Class for handling sentiment analysis with different model types"""
    
    def __init__(self, model_type=None, model_path=None):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_type: Type of model ('original', 'enhanced', or 'balanced')
            model_path: Path to a specific model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load dataset for tokenizer initialization
        print("Loading dataset to prepare tokenizer...")
        try:
            data = load_data("data\\final_dataset.json")
            
            # Initialize tokenizer with emoji support
            print("Preparing tokenizer with emoji support...")
            self.tokenizer, num_emoji_tokens = prepare_tokenizer(data)
            print(f"Tokenizer prepared with {num_emoji_tokens} emoji tokens")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using default BERT tokenizer without emoji support")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load model
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self._load_model()
    def _find_best_model(self):
        """Find the best model based on the requested type"""
        print(f"Looking for {self.model_type if self.model_type else 'best'} model...")
        
        # Find appropriate run directories
        run_dirs = sorted([os.path.join("runs", d) for d in os.listdir("runs") 
                           if os.path.isdir(os.path.join("runs", d))], 
                         key=os.path.getmtime)
        
        if not run_dirs:
            return None
            
        # For enhanced/balanced models, we need to be smarter about detection
        # Since we don't have directory naming conventions, we'll check based on dates
        # Original model is likely the earliest one, enhanced models are newer
        if self.model_type == 'original':
            # Original model is likely the earliest one
            # Take the first run directory (earliest by timestamp)
            candidate_runs = [run_dirs[0]] if run_dirs else []
            print(f"Using earliest run as original model: {candidate_runs[0] if candidate_runs else 'None'}")
        elif self.model_type in ('enhanced', 'balanced'):
            # Enhanced/balanced models are likely more recent
            # Take any run directory except the first one (which is the original)
            candidate_runs = run_dirs[1:] if len(run_dirs) > 1 else run_dirs
            print(f"Found {len(candidate_runs)} candidate runs for enhanced model")
        else:
            # Use all runs as candidates, prioritizing newer ones
            candidate_runs = run_dirs
          # Search for best model in candidate runs (newest first)
        for run_dir in reversed(candidate_runs):
            print(f"Checking run directory: {run_dir}")
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                print(f"No checkpoints directory in {run_dir}")
                continue
                
            # List files in checkpoint directory
            checkpoint_files = os.listdir(checkpoint_dir)
            if not checkpoint_files:
                print(f"Checkpoints directory is empty in {run_dir}")
                continue
                
            print(f"Found checkpoint files: {checkpoint_files}")
            
            # Try to find best model checkpoints
            best_models = [f for f in checkpoint_files if f.startswith('best_model')]
            if best_models:
                # Sort by F1 score in filename
                try:
                    best_model = sorted(best_models, 
                                      key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), 
                                      reverse=True)[0]
                    model_path = os.path.join(checkpoint_dir, best_model)
                    print(f"Found best model: {best_model} with path: {model_path}")
                    return model_path
                except Exception as e:
                    print(f"Error parsing model filenames: {e}")
            else:
                # If no best_model files found, try latest_model.pt as fallback
                if 'latest_model.pt' in checkpoint_files:
                    model_path = os.path.join(checkpoint_dir, 'latest_model.pt')
                    print(f"No best_model found, using latest_model.pt instead: {model_path}")
                    return model_path
        
        # Check backup directory as a last resort
        backup_dir = "model_backups"
        if os.path.exists(backup_dir):
            print("Checking model_backups directory...")
            backup_dirs = sorted([os.path.join(backup_dir, d) for d in os.listdir(backup_dir) 
                               if os.path.isdir(os.path.join(backup_dir, d))],
                              key=os.path.getmtime, reverse=True)
            
            for backup in backup_dirs:
                # Look for model directories within each backup
                model_dirs = [os.path.join(backup, d) for d in os.listdir(backup)
                           if os.path.isdir(os.path.join(backup, d))]
                
                for model_dir in model_dirs:
                    checkpoint_dir = os.path.join(model_dir, "checkpoints")
                    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
                        best_models = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model')]
                        if best_models:
                            best_model = sorted(best_models, 
                                             key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), 
                                             reverse=True)[0]
                            model_path = os.path.join(checkpoint_dir, best_model)
                            print(f"Found model in backup: {model_path}")
                            return model_path
        
        print("No suitable model checkpoint found.")
        return None
    def _load_model(self):
        """Load the appropriate model based on type or path"""
        if not self.model_path:
            self.model_path = self._find_best_model()
            if not self.model_path:
                # Try finding a model from the most recent run directory as a last resort
                run_dirs = sorted([os.path.join("runs", d) for d in os.listdir("runs") 
                                  if os.path.isdir(os.path.join("runs", d))],
                                 key=os.path.getmtime, reverse=True)
                
                if run_dirs:
                    latest_run = run_dirs[0]
                    print(f"Trying latest run directory as fallback: {latest_run}")
                    
                    # Look for any model file in the checkpoints directory
                    checkpoint_dir = os.path.join(latest_run, "checkpoints")
                    if os.path.exists(checkpoint_dir):
                        model_files = os.listdir(checkpoint_dir)
                        if model_files:
                            # Pick first available model file
                            self.model_path = os.path.join(checkpoint_dir, model_files[0])
                            print(f"Found model file as fallback: {self.model_path}")
                
                # If still no model found, check if we need to train one
                if not self.model_path:
                    print("No model found. Please train a model first using train.py or train_enhanced.py")
                    raise ValueError("No suitable model checkpoint found")
                
        print(f"Loading model from: {self.model_path}")
        
        # Try loading with different model classes based on type
        if self.model_type == 'original':
            try:
                self.model, _ = load_model(self.model_path, SentimentModel, self.tokenizer, self.device)
                print("Loaded original sentiment model")
                return
            except Exception as e:
                print(f"Failed to load original model: {e}")
                
        elif self.model_type == 'enhanced':
            try:
                self.model, _ = load_model(self.model_path, EnhancedSentimentModel, self.tokenizer, self.device)
                print("Loaded enhanced sentiment model")
                return
            except Exception as e:
                print(f"Failed to load enhanced model: {e}")
                
        elif self.model_type == 'balanced':
            try:
                self.model, _ = load_model(self.model_path, BalancedSentimentModel, self.tokenizer, self.device)
                print("Loaded balanced sentiment model")
                return
            except Exception as e:
                print(f"Failed to load balanced model: {e}")
                
        # Auto-detect model type if not specified or failed to load with specified type
        model_classes = [BalancedSentimentModel, EnhancedSentimentModel, SentimentModel]
        model_names = ["Balanced", "Enhanced", "Original"]
        
        for cls, name in zip(model_classes, model_names):
            try:
                self.model, _ = load_model(self.model_path, cls, self.tokenizer, self.device)
                print(f"Successfully loaded model as {name} model")
                return
            except Exception as e:
                print(f"Failed to load as {name} model: {e}")
        
        raise ValueError("Could not load model with any available model class")
    
    def analyze(self, text, max_length=512):
        """Analyze sentiment of the given text"""
        if not self.model:
            raise ValueError("Model not loaded")
            
        return predict_sentiment(text, self.model, self.tokenizer, self.device, max_length)

if __name__ == "__main__":
    # Parse command arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis with Enhanced Models')
    parser.add_argument('--model-type', type=str, choices=['original', 'enhanced', 'balanced'],
                       help='Type of model to use for prediction')
    parser.add_argument('--model-path', type=str, help='Path to a specific model checkpoint')
    parser.add_argument('--text', type=str, help='Text to analyze (in quotes)')
    parser.add_argument('--file', type=str, help='File containing texts to analyze (one per line)')
    args = parser.parse_args()
    
    try:
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer(model_type=args.model_type, model_path=args.model_path)
        
        # Process file if provided
        if args.file and os.path.exists(args.file):
            print(f"\nAnalyzing texts from file: {args.file}")
            with open(args.file, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    
                    try:
                        result = analyzer.analyze(text)
                        print(f"\nText: {text}")
                        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
                        print(f"Probabilities: Negative: {result['probabilities']['Negative']:.2f}%, " +
                             f"Neutral: {result['probabilities']['Neutral']:.2f}%, " +
                             f"Positive: {result['probabilities']['Positive']:.2f}%")
                    except Exception as e:
                        print(f"\nError processing: {text}")
                        print(f"Error details: {e}")
        
        # Process single text if provided
        elif args.text:
            try:
                result = analyzer.analyze(args.text)
                print(f"\nText: {args.text}")
                print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
                print(f"Probabilities: Negative: {result['probabilities']['Negative']:.2f}%, " +
                     f"Neutral: {result['probabilities']['Neutral']:.2f}%, " +
                     f"Positive: {result['probabilities']['Positive']:.2f}%")
            except Exception as e:
                print(f"\nError processing: {args.text}")
                print(f"Error details: {e}")
        
        # Run interactive mode with examples if no text/file provided
        else:
            # Test with some example sentences
            test_sentences = [
                "I am really happy with this product!",
                "This movie was terrible and a waste of time.",
                "The service was okay, nothing special.",
                "I love this new feature, it's amazing! 😍",
                "This is the worst experience I've ever had 😠"
            ]
            
            print("\n===== SENTIMENT PREDICTIONS =====")
            for sentence in test_sentences:
                try:
                    result = analyzer.analyze(sentence)
                    print(f"\nText: {sentence}")
                    print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
                    print(f"Probabilities: Negative: {result['probabilities']['Negative']:.2f}%, " +
                         f"Neutral: {result['probabilities']['Neutral']:.2f}%, " +
                         f"Positive: {result['probabilities']['Positive']:.2f}%")
                except Exception as e:
                    print(f"\nError processing: {sentence}")
                    print(f"Error details: {e}")
            
            print("\nEnter your own text for sentiment analysis (type 'exit' to quit):")
            while True:
                user_input = input("> ")
                if user_input.lower() == 'exit':
                    break
                    
                try:
                    result = analyzer.analyze(user_input)
                    print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
                    print(f"Probabilities: Negative: {result['probabilities']['Negative']:.2f}%, " +
                         f"Neutral: {result['probabilities']['Neutral']:.2f}%, " +
                         f"Positive: {result['probabilities']['Positive']:.2f}%")
                except Exception as e:
                    print(f"Error processing your input: {e}")
    
    except Exception as e:
        print(f"Error initializing sentiment analyzer: {e}")

import torch
import argparse
import os
from transformers import BertTokenizer
from load_sentiment_model import create_model_from_checkpoint, predict_sentiment

def main():
    """
    Interactive sentiment analysis demo script
    """
    parser = argparse.ArgumentParser(description='Interactive Sentiment Analysis')
    parser.add_argument('--model-path', type=str, default='runs/custom_model/kaggle2.pt', 
                        help='Path to the model weights file')
    parser.add_argument('--config-path', type=str, default='runs/custom_model/config.json', 
                        help='Path to the model config file')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        return
    
    # Create model and load weights with correct architecture detection
    print("\n===== LOADING SENTIMENT MODEL =====")
    print(f"Model path: {args.model_path}")
    print(f"Config path: {args.config_path}")
    
    try:
        model, tokenizer = create_model_from_checkpoint(args.config_path, args.model_path, device)
        print("\n✅ Model loaded successfully! Ready for sentiment analysis.")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Interactive mode
    print("\n===== SENTIMENT ANALYSIS DEMO =====")
    print("Enter text for sentiment analysis.")
    print("Type 'examples' to see example sentences.")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "I'm not sure how I feel about this product. It has pros and cons.",
        "The customer service was terrible and I'll never shop here again.",
        "The weather today is quite pleasant.",
        "I was disappointed by the quality of this item."
    ]
    
    history = []
    
    while True:
        print()
        user_input = input("> ")
        
        if user_input.lower() == 'exit':
            print("Thank you for using the sentiment analysis tool!")
            break
            
        elif user_input.lower() == 'examples':
            print("\n----- Example Sentences -----")
            for i, example in enumerate(examples):
                print(f"{i+1}. {example}")
            print("-" * 50)
            continue
            
        elif user_input.lower() == 'history':
            if not history:
                print("No analysis history yet.")
            else:
                print("\n----- Analysis History -----")
                for i, (text, result) in enumerate(history):
                    print(f"{i+1}. \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
                    print(f"   Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
                print("-" * 50)
            continue
        
        elif not user_input.strip():
            continue
        
        # Process the input
        result = predict_sentiment(user_input, model, tokenizer, device)
        
        # Format the output
        print(f"\nSentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nProbabilities:")
        print(f"  Negative: {result['probabilities']['Negative']:.2f}%")
        print(f"  Neutral: {result['probabilities']['Neutral']:.2f}%") 
        print(f"  Positive: {result['probabilities']['Positive']:.2f}%")
        
        # Add to history
        history.append((user_input, result))
        
        # Visual separator
        print("-" * 50)

if __name__ == "__main__":
    main()

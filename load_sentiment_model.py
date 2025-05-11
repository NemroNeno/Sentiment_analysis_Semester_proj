import torch
import json
import os
import argparse
from transformers import BertTokenizer
from enhanced_models import BalancedSentimentModel

def load_model_config(config_path):
    """Load model configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded model config from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def detect_model_architecture(weights_path, device='cpu'):
    """Analyze checkpoint to detect model architecture parameters"""
    print(f"Analyzing model architecture from {weights_path}")
    
    # Add numpy to safe globals for torch.load
    import torch.serialization
    try:
        import numpy
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        print("Added numpy.core.multiarray.scalar to safe globals")
    except Exception as e:
        print(f"Warning: Could not add numpy to safe globals: {e}")
    
    try:
        # First try with weights_only=False (required for older models)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        # Extract state dict if checkpoint has multiple components
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Detect architecture parameters from checkpoint
        architecture = {}
        
        # 1. Detect number of LSTM layers
        lstm_layer_keys = [k for k in state_dict.keys() if k.startswith('lstm.weight_ih_l') or k.startswith('lstm.weight_hh_l')]
        max_layer_num = -1
        for key in lstm_layer_keys:
            # Extract the layer number from keys like 'lstm.weight_ih_l0', 'lstm.weight_hh_l1'
            try:
                layer_num = int(key.split('_l')[-1][0])
                max_layer_num = max(max_layer_num, layer_num)
            except (IndexError, ValueError):
                continue
                
        if max_layer_num >= 0:
            actual_lstm_layers = max_layer_num + 1
            architecture['num_lstm_layers'] = actual_lstm_layers
            print(f"Detected {actual_lstm_layers} LSTM layers in checkpoint")
        
        # 2. Detect attention heads (if possible)
        attention_keys = [k for k in state_dict.keys() if 'attention' in k and '.query.' in k]
        if attention_keys:
            for key in attention_keys:
                if hasattr(state_dict[key], 'shape') and len(state_dict[key].shape) == 2:
                    hidden_dim = state_dict[key].shape[0]
                    # Try to infer num_heads
                    for num_heads in [4, 8, 12, 16]:  # Common values
                        if hidden_dim % num_heads == 0:
                            architecture['num_attention_heads'] = num_heads
                            print(f"Detected {num_heads} attention heads")
                            break
                    break
        
        # 3. Check for class weights
        if any('class_weights' in k for k in state_dict.keys()):
            architecture['has_class_weights'] = True
            print("Model has class weights")
            
            # Try to extract actual class weights if available
            class_weight_keys = [k for k in state_dict.keys() if 'class_weights' in k]
            if class_weight_keys:
                weights = state_dict[class_weight_keys[0]].tolist()
                architecture['class_weights'] = weights
                print(f"Detected class weights: {weights}")
        
        return architecture, state_dict
    
    except Exception as e:
        print(f"Error analyzing model architecture: {e}")
        return {}, None

def create_model_from_checkpoint(config_path, weights_path, device='cpu'):
    """Create a model with the correct architecture based on checkpoint analysis"""
    # First load the config file
    config = load_model_config(config_path)
    
    # Then analyze the checkpoint to determine architecture
    architecture, _ = detect_model_architecture(weights_path, device)
    
    # Update config with detected architecture parameters
    if 'num_lstm_layers' in architecture:
        config['num_lstm_layers'] = architecture['num_lstm_layers']
        print(f"Using {architecture['num_lstm_layers']} LSTM layers from checkpoint")
    
    if 'num_attention_heads' in architecture:
        config['num_attention_heads'] = architecture['num_attention_heads']
        print(f"Using {architecture['num_attention_heads']} attention heads from checkpoint")
    
    if 'class_weights' in architecture:
        config['class_weights'] = architecture['class_weights']
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer)
    
    # Extract model parameters from config
    embedding_dim = config.get('embedding_dim', 768)
    hidden_dim = config.get('hidden_dim', 256)
    output_dim = config.get('output_dim', 3)
    num_lstm_layers = config.get('num_lstm_layers', 2)  # Default to 2 if not specified
    num_attention_heads = config.get('num_attention_heads', 8)
    dropout = config.get('dropout', 0.3)
    class_weights = config.get('class_weights', None)
    
    # Show model creation details
    print(f"\nCreating BalancedSentimentModel with parameters:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - hidden_dim: {hidden_dim}")
    print(f"  - output_dim: {output_dim}")
    print(f"  - num_lstm_layers: {num_lstm_layers}")
    print(f"  - num_attention_heads: {num_attention_heads}")
    print(f"  - dropout: {dropout}")
    if class_weights:
        print(f"  - class_weights: {class_weights}")
    
    # Create model with correct architecture
    model = BalancedSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_lstm_layers=num_lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        class_weights=class_weights
    )
    
    # Load weights
    print(f"Loading weights from {weights_path}")
    try:
        # Add numpy to safe globals for torch.load
        import torch.serialization
        try:
            import numpy
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        except Exception:
            pass
            
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        # Extract state dict if checkpoint has multiple components
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Track key loading statistics
        model_dict = model.state_dict()
        compatible_keys = [k for k in state_dict.keys() if k in model_dict]
        missing_keys = [k for k in model_dict.keys() if k not in state_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in model_dict]
        
        # Filter compatible keys
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        # Handle embedding size mismatch
        if 'embedding.weight' in filtered_dict and 'embedding.weight' in model_dict:
            saved_embeds = filtered_dict['embedding.weight']
            current_embeds = model_dict['embedding.weight']
            
            if saved_embeds.size(0) != current_embeds.size(0):
                print(f"Embedding size mismatch: saved={saved_embeds.size(0)}, current={current_embeds.size(0)}")
                min_vocab_size = min(saved_embeds.size(0), current_embeds.size(0))
                filtered_dict['embedding.weight'] = saved_embeds[:min_vocab_size]
        
        # Load state dict
        model.load_state_dict(filtered_dict, strict=False)
        
        # Print detailed loading statistics
        print(f"Loaded {len(compatible_keys)}/{len(model_dict)} parameters from checkpoint")
        
        # Report missing keys in a structured way
        if missing_keys:
            print(f"Missing {len(missing_keys)} keys:")
            key_types = {
                'lstm': [k for k in missing_keys if k.startswith('lstm.')],
                'attention': [k for k in missing_keys if 'attention' in k],
                'classifier': [k for k in missing_keys if 'classifier' in k],
                'other': [k for k in missing_keys if not (k.startswith('lstm.') or 'attention' in k or 'classifier' in k)]
            }
            
            for key_type, keys in key_types.items():
                if keys:
                    print(f"  - {key_type}: {len(keys)} keys")
            
        # Move model to the specified device
        model = model.to(device)
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise

def predict_sentiment(text, model, tokenizer, device='cpu', max_length=512):
    """Make a sentiment prediction for the given text"""
    # Prepare model for inference
    model.eval()
    
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
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, prediction = torch.max(probabilities, dim=0)
    
    # Map prediction index to sentiment label
    sentiment_map = {
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    }
    
    sentiment = sentiment_map.get(prediction.item(), 'Unknown')
    confidence_pct = confidence.item() * 100
    
    # Return prediction results
    return {
        'sentiment': sentiment,
        'confidence': confidence_pct,
        'probabilities': {
            'Negative': probabilities[0].item() * 100,
            'Neutral': probabilities[1].item() * 100,
            'Positive': probabilities[2].item() * 100
        }
    }

def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Sentiment Analysis with Correct Model Loading')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--model-path', type=str, default='runs/custom_model/kaggle2.pt', 
                        help='Path to the model weights file')
    parser.add_argument('--config-path', type=str, default='runs/custom_model/config.json', 
                        help='Path to the model config file')
    parser.add_argument('--lstm-layers', type=int, help='Override: Number of LSTM layers to use in the model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get model paths
    model_path = args.model_path
    config_path = args.config_path
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    # Create model and load weights with correct architecture detection
    model, tokenizer = create_model_from_checkpoint(config_path, model_path, device)
    
    # Override with command line argument if provided
    if args.lstm_layers is not None:
        print(f"Warning: Manual override of LSTM layers to {args.lstm_layers}. This may cause loading issues.")
    
    # Run interactive mode or single text mode
    if args.text:
        # Single text mode
        result = predict_sentiment(args.text, model, tokenizer, device)
        print(f"\nText: {args.text}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
        print(f"Probabilities: Negative: {result['probabilities']['Negative']:.2f}%, " +
             f"Neutral: {result['probabilities']['Neutral']:.2f}%, " +
             f"Positive: {result['probabilities']['Positive']:.2f}%")
    else:
        # Interactive mode
        print("\n===== SENTIMENT ANALYSIS =====")
        print("Enter your text for sentiment analysis (type 'exit' to quit):")
        
        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                break
            
            result = predict_sentiment(user_input, model, tokenizer, device)
            print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
            print(f"Probabilities: Negative: {result['probabilities']['Negative']:.2f}%, " +
                 f"Neutral: {result['probabilities']['Neutral']:.2f}%, " +
                 f"Positive: {result['probabilities']['Positive']:.2f}%")

if __name__ == "__main__":
    main()

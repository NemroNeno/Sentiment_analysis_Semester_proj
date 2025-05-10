import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Import from our modules
from dataset import set_seed, load_data, prepare_tokenizer, create_data_loaders
from enhanced_models import EnhancedSentimentModel, BalancedSentimentModel
from utils import init_model_with_bert_embeddings, visualize_results
from train_utils import train_model

# Set seed for reproducibility
set_seed(42)

def calculate_class_weights(data):
    """Calculate class weights to address class imbalance"""
    labels = [item['polarity'] + 1 for item in data]  # Convert to 0, 1, 2
    label_counts = Counter(labels)
    n_samples = len(data)
    n_classes = 3  # Negative, Neutral, Positive
      # Calculate weights as inverse of frequency
    weights = [n_samples / (n_classes * label_counts[i]) for i in range(n_classes)]
    
    # Normalize weights so they sum to n_classes
    weights = np.array(weights, dtype=np.float32)  # Explicitly use float32
    weights = weights / weights.sum() * n_classes
    
    print(f"Class distribution: {label_counts}")
    print(f"Calculated weights: {weights}")
    
    return weights

if __name__ == "__main__":
    print("Loading data...")
    # Load and preprocess data
    data = load_data(f"data\\final_dataset.json")
    
    print("Preparing tokenizer...")
    # Initialize tokenizer with emoji support
    bert_tokenizer, num_emoji_tokens = prepare_tokenizer(data)
    
    print("Creating data loaders...")
    # Create train and test data loaders
    train_loader, test_loader = create_data_loaders(data, bert_tokenizer, batch_size=16)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Calculate class weights to address class imbalance
    class_weights = calculate_class_weights(data)
    
    # Define model architecture parameters
    model_params = {
        'vocab_size': len(bert_tokenizer),
        'embedding_dim': 768,  # BERT embedding dimension
        'hidden_dim': 256,     # LSTM hidden dimension
        'output_dim': 3,       # 3 sentiment classes
        'num_lstm_layers': 2,  # Number of LSTM layers
        'num_attention_heads': 4,  # Number of attention heads
        'dropout': 0.3,        # Dropout rate
        'class_weights': class_weights  # Class weights
    }
    
    print("Initializing enhanced model with balanced class weights...")
    model = BalancedSentimentModel(**model_params)
    model.to(device)
    
    # Initialize model with BERT embeddings
    print("Loading BERT embeddings...")
    model = init_model_with_bert_embeddings(model, bert_tokenizer, device)
      # Create custom loss function with explicit typing
    def weighted_loss_fn(outputs, labels):
        # Instead of using model's get_loss, implement the weighted loss directly
        weights = model.class_weights
        return nn.functional.cross_entropy(outputs, labels, weight=weights)
    
    # Setup training with a lower learning rate for stability
    criterion = weighted_loss_fn  # Use weighted loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # AdamW with weight decay
    
    # Create directories for runs if they don't exist
    os.makedirs("runs", exist_ok=True)
      # Train model with weighted loss
    print("Starting enhanced model training...")
    try:
        trained_model, best_f1, run_dir = train_model(
            model, 
            train_loader, 
            test_loader, 
            criterion, 
            optimizer,
            device,
            num_epochs=15,  # Train for more epochs
            eval_every=1
        )
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()
        print("\nFalling back to direct training without train_model function...")
        
        # If train_model fails, implement direct training here
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("runs", f"sentiment_model_{timestamp}")
        model_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        
        # Train for 3 epochs as a test and save the model
        model.train()
        for epoch in range(3):
            print(f"Manual training: Epoch {epoch+1}/3")
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                
                # Use direct loss calculation with explicit type casting if needed
                loss = nn.functional.cross_entropy(
                    outputs, 
                    labels,
                    weight=model.class_weights
                )
                
                loss.backward()
                optimizer.step()
                
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'model_class': model.__class__.__name__,
                'embedding_dim': model.embedding.embedding_dim,
                'hidden_dim': model.lstm.hidden_size
            }
        }, os.path.join(model_dir, "manual_trained_model.pt"))
        
        best_f1 = 0  # Placeholder
    
    # Visualize results from the training run
    print("Generating visualizations...")
    visualize_results(run_dir, BalancedSentimentModel, bert_tokenizer, device)
    
    print("\n=== Enhanced Model Training Complete ===")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"TensorBoard logs and model checkpoints saved in: {run_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={run_dir}")
    print("=========================================")

import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import from our modules
from dataset import set_seed, load_data, prepare_tokenizer, create_data_loaders
from models import SentimentModel
from utils import init_model_with_bert_embeddings, visualize_results
from train_utils import train_model

# Set seed for reproducibility
set_seed(42)

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
    
    # Initialize model
    print("Initializing model...")
    model = SentimentModel(vocab_size=len(bert_tokenizer), embedding_dim=768, hidden_dim=256, output_dim=3)
    model.to(device)
    
    # Initialize model with BERT embeddings
    print("Loading BERT embeddings...")
    model = init_model_with_bert_embeddings(model, bert_tokenizer, device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
    # Create directories for runs if they don't exist
    os.makedirs("runs", exist_ok=True)
    
    # Train model
    print("Starting training...")
    trained_model, best_f1, run_dir = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer,
        device,
        num_epochs=10, 
        eval_every=1
    )
    
    # Visualize results from the training run
    print("Generating visualizations...")
    visualize_results(run_dir, SentimentModel, bert_tokenizer, device)
    
    print("\n=== Model Training Complete ===")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"TensorBoard logs and model checkpoints saved in: {run_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={run_dir}")
    print("================================")
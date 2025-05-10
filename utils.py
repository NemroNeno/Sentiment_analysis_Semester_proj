import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
import numpy as np  # Explicitly import numpy
import os
import datetime
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def init_model_with_bert_embeddings(model, bert_tokenizer, device):
    """Initialize model's embedding layer with pretrained BERT embeddings"""
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    with torch.no_grad():
        # Copy BERT's pretrained embeddings for the original vocabulary
        # Ensure the embedding weights have the correct dtype (float32)
        bert_embeddings = bert_model.embeddings.word_embeddings.weight.to(device, dtype=torch.float32)
        model.embedding.weight[:bert_tokenizer.vocab_size] = bert_embeddings
        
        # Initialize embeddings for new emoji tokens with random values
        num_new_tokens = len(bert_tokenizer) - bert_tokenizer.vocab_size
        if num_new_tokens > 0:
            new_embeddings = torch.randn(num_new_tokens, 768, dtype=torch.float32, device=device) * 0.02
            model.embedding.weight[bert_tokenizer.vocab_size:] = new_embeddings
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test data"""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Create a progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc="Evaluating", total=len(test_loader))
    
    with torch.no_grad():
        for batch in eval_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive'])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print('\nClassification Report:')
    print(report)
    
    # Return metrics dictionary for logging
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
    
    return metrics

def load_model(model_path, model_class, tokenizer, device):
    """Load a saved model from a checkpoint"""
    # Handle pickle error with NumPy arrays in PyTorch 2.6+
    import numpy as np
    
    try:
        # First try with weights_only=False
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e1:
        try:
            # If that fails, add NumPy to safe globals and try again
            torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e2:
            try:
                # Last resort: try with a context manager
                with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct']):
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            except Exception as e3:
                print(f"Error loading model: {e3}")
                print("Trying alternative approach...")
                # As a last resort - recreate the model and just load the state dict
                temp_model = model_class(vocab_size=len(tokenizer), embedding_dim=768, hidden_dim=256, output_dim=3)
                checkpoint = {'model_state_dict': torch.load(model_path, map_location=device, weights_only=True)}
    
    config = checkpoint.get('config', {})
      # Extract model parameters from config or use defaults
    embedding_dim = config.get('embedding_dim', 768)
    hidden_dim = config.get('hidden_dim', 256)
    output_dim = config.get('output_dim', 3)
    dropout = config.get('dropout', 0.3)
    
    # Check model_class and configure parameters accordingly
    if model_class.__name__ == 'BalancedSentimentModel':
        # Initialize with default class weights if not in config
        class_weights = config.get('class_weights', [1.0, 1.0, 1.0])
        model = model_class(
            vocab_size=len(tokenizer), 
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            dropout=dropout,
            class_weights=class_weights
        )
    else:
        # Initialize model with the same architecture
        model = model_class(
            vocab_size=len(tokenizer), 
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            dropout=dropout
        )
    model.to(device)
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Handle printing of metrics with proper formatting
    loss = checkpoint.get('loss')
    f1_score = checkpoint.get('f1_score')
    accuracy = checkpoint.get('accuracy')
    
    loss_str = f"{loss:.4f}" if isinstance(loss, float) else "N/A"
    f1_str = f"{f1_score:.4f}" if isinstance(f1_score, float) else "N/A"
    acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else "N/A"
    
    print(f"Model metrics - Loss: {loss_str}, F1 Score: {f1_str}, Accuracy: {acc_str}")
    
    return model, checkpoint

def visualize_results(run_dir, model_class, tokenizer, device):
    """Generate and save visualizations of results after training"""
    # Find the best model checkpoint
    model_dir = os.path.join(run_dir, "checkpoints")
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
    
    if not checkpoints:
        print("No best model checkpoint found.")
        return
    
    # Sort by F1 score in the filename
    best_checkpoint = sorted(checkpoints, key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), reverse=True)[0]
    best_model_path = os.path.join(model_dir, best_checkpoint)
    
    # Load best model and checkpoint data
    model, checkpoint = load_model(best_model_path, model_class, tokenizer, device)
      # If the confusion matrix was saved in the checkpoint, visualize it
    if 'confusion_matrix' in checkpoint:
        conf_matrix = checkpoint['confusion_matrix']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix (Best Model)')
        plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), bbox_inches='tight')
        plt.close(fig)
    
    # Create a summary report
    with open(os.path.join(run_dir, "results_summary.txt"), 'w') as f:
        f.write("===== SENTIMENT ANALYSIS MODEL RESULTS =====\n\n")
        f.write(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best model: {best_checkpoint}\n")
        
        # Format metrics properly for writing to file
        f1_score = checkpoint.get('f1_score')
        f1_str = f"{f1_score:.4f}" if isinstance(f1_score, float) else str(f1_score) if f1_score is not None else "N/A"
        
        accuracy = checkpoint.get('accuracy')
        acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy) if accuracy is not None else "N/A"
        
        f.write(f"F1 Score: {f1_str}\n")
        f.write(f"Accuracy: {acc_str}\n\n")
        
        if 'classification_report' in checkpoint:
            f.write("Classification Report:\n")
            f.write(checkpoint['classification_report'])
    
    print(f"Results visualizations and summary saved to {run_dir}")

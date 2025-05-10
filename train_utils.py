import torch
import os
import datetime
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from utils import evaluate_model

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10, eval_every=1):
    """
    Training loop with TensorBoard logging and best model saving
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to run training on (cuda/cpu)
        num_epochs: Number of epochs to train for
        eval_every: Evaluate on test set every n epochs
    
    Returns:
        model: Trained model
        best_f1: Best F1 score achieved during training
    """
    # Create a unique run directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"sentiment_model_{timestamp}")
    model_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=run_dir)
      # Save training config - handle different model architectures
    config = {
        "embedding_dim": model.embedding.embedding_dim,
        "hidden_dim": model.lstm.hidden_size,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_loader.batch_size,
        "num_epochs": num_epochs,
        "tokenizer": "bert-base-uncased",
        "model_class": model.__class__.__name__
    }
    
    # Add model-specific attributes
    if hasattr(model, 'fc'):
        config["output_dim"] = model.fc.out_features
    elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        # Get the last layer of the sequential classifier
        last_layer = list(model.classifier.children())[-1]
        config["output_dim"] = last_layer.out_features
    else:
        config["output_dim"] = 3  # Default for sentiment analysis
    
    # Add dropout if available
    if hasattr(model, 'dropout'):
        config["dropout"] = model.dropout.p
    
    # Add enhanced model specific features
    if hasattr(model, 'attention') and hasattr(model.attention, 'num_heads'):
        config["num_attention_heads"] = model.attention.num_heads
    
    if hasattr(model, 'class_weights'):
        config["has_class_weights"] = True
    
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    model.train()
    best_f1 = 0.0  # Track best F1 score for model saving
    
    # Create a progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create a progress bar for batches within the epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                          leave=False, position=1, total=len(train_loader))
        
        for batch_idx, batch in enumerate(batch_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            batch_acc = correct / total
            
            total_loss += loss.item()
            
            # Update the batch progress bar with current loss and accuracy
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})
            
            # Log batch metrics
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/batch_accuracy', batch_acc, global_step)
        
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Log epoch metrics
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_accuracy', train_acc, epoch)
        
        # Update the epoch progress bar with average loss and accuracy
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "acc": f"{train_acc:.4f}"})
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}')
        
        # Evaluate on test set every eval_every epochs or on the last epoch
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            print(f"\nEvaluating at epoch {epoch+1}...")
            metrics = evaluate_model(model, test_loader, device)
            
            # Log evaluation metrics
            writer.add_scalar('eval/accuracy', metrics['accuracy'], epoch)
            writer.add_scalar('eval/f1_score', metrics['f1_score'], epoch)
            
            # Create and log confusion matrix figure
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Neutral', 'Positive'],
                       yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
            writer.add_figure('eval/confusion_matrix', fig, epoch)
            plt.close(fig)
            
            # Save model if it's the best so far based on F1 score
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}_f1_{best_f1:.4f}.pt")
                
                # Save model with metadata
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'config': config,
                    'classification_report': metrics['classification_report'],
                    'confusion_matrix': metrics['confusion_matrix']
                }, model_path)
                
                print(f"🔥 New best model saved with F1: {best_f1:.4f}")
                
            # Always save the latest model
            latest_model_path = os.path.join(model_dir, "latest_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, latest_model_path)
    
    # Log hyperparameters and final metrics
    writer.add_hparams(
        hparam_dict=config,
        metric_dict={
            'hparam/best_f1_score': best_f1,
            'hparam/final_train_loss': avg_loss,
            'hparam/final_train_accuracy': train_acc
        }
    )
    
    writer.close()
    print(f"\nTraining completed! Best F1 Score: {best_f1:.4f}")
    print(f"Models and logs saved to: {run_dir}")
    
    return model, best_f1, run_dir

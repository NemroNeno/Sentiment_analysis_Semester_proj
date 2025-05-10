import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from dataset import load_data, prepare_tokenizer, create_data_loaders
from models import SentimentModel
from enhanced_models import EnhancedSentimentModel, BalancedSentimentModel
from utils import load_model

def compare_models(models_info, test_loader, device):
    """Compare multiple models on the same test data"""
    results = {}
    
    for name, (model, model_class) in models_info.items():
        print(f"\nEvaluating {name}...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
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
        report = classification_report(all_labels, all_preds, 
                                      target_names=['Negative', 'Neutral', 'Positive'])
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print('\nClassification Report:')
        print(report)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    return results

def plot_comparison(results, save_path='model_comparison.png'):
    """Plot comparison metrics between models"""
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracies
    sns.barplot(x=model_names, y=accuracies, palette='viridis', ax=ax1)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # Plot F1 scores
    sns.barplot(x=model_names, y=f1_scores, palette='viridis', ax=ax2)
    ax2.set_title('F1 Score Comparison')
    ax2.set_ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(results, save_dir='.'):
    """Plot confusion matrices for all models"""
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()

def analyze_errors(results, data_loader, tokenizer):
    """Analyze error patterns across models"""
    # Collect prediction disagreements
    models = list(results.keys())
    all_examples = []
    
    # Get indices where at least one model is wrong
    true_labels = results[models[0]]['true_labels']
    error_indices = set()
    
    for name in models:
        preds = results[name]['predictions']
        for i, (pred, true) in enumerate(zip(preds, true_labels)):
            if pred != true:
                error_indices.add(i)
    
    # Convert dataset to list for easy access
    dataset = []
    for batch in data_loader:
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]
        labels = batch['label'].tolist()
        for text, label in zip(texts, labels):
            dataset.append({'text': text, 'label': label})
    
    # Collect prediction details for error cases
    error_analysis = []
    
    for idx in error_indices:
        if idx >= len(dataset):
            continue
            
        example = {
            'text': dataset[idx]['text'],
            'true_label': true_labels[idx],
            'true_sentiment': ['Negative', 'Neutral', 'Positive'][true_labels[idx]],
            'predictions': {}
        }
        
        for name in models:
            pred = results[name]['predictions'][idx]
            example['predictions'][name] = {
                'predicted_label': int(pred),
                'predicted_sentiment': ['Negative', 'Neutral', 'Positive'][pred],
                'correct': pred == true_labels[idx]
            }
        
        error_analysis.append(example)
    
    # Write error analysis to file
    with open('model_error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=4)
    
    return error_analysis

def main():
    # Load data
    print("Loading data...")
    data = load_data(f"data\\final_dataset.json")
    
    # Initialize tokenizer
    print("Preparing tokenizer...")
    bert_tokenizer, num_emoji_tokens = prepare_tokenizer(data)
    
    # Create data loaders - we only need the test loader
    print("Creating data loaders...")
    _, test_loader = create_data_loaders(data, bert_tokenizer, batch_size=16)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all run directories
    run_dirs = sorted([os.path.join("runs", d) for d in os.listdir("runs")], key=os.path.getmtime)
    
    if len(run_dirs) < 2:
        print("Need at least two model runs to compare.")
        return
      # Find runs with valid models
    valid_runs = []
    
    for run_dir in run_dirs:
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            continue
            
        best_models = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model')]
        if best_models:
            valid_runs.append(run_dir)
    
    if len(valid_runs) < 2:
        print("Could not find at least two valid model runs with best_model checkpoints.")
        print("Available runs:", run_dirs)
        
        # Try to find the original model from before we started enhanced training
        original_model_dir = "runs/sentiment_model_20250510-000044"
        if os.path.exists(os.path.join(original_model_dir, "checkpoints")):
            print(f"Found original model at {original_model_dir}, will use this for comparison.")
            original_run = original_model_dir
            enhanced_run = valid_runs[-1] if valid_runs else None
        else:
            print("Please train both original and enhanced models first.")
            return
    else:
        # Use the two most recent valid runs
        original_run = valid_runs[-2]  # Second-to-last valid run
        enhanced_run = valid_runs[-1]  # Most recent valid run
    
    print(f"Original model run: {original_run}")
    print(f"Enhanced model run: {enhanced_run}")
    
    # Load best models from both runs
    models_info = {}
    
    # Load original model
    model_dir = os.path.join(original_run, "checkpoints")
    best_models = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
    
    if best_models:
        best_model = sorted(best_models, key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), reverse=True)[0]
        best_model_path = os.path.join(model_dir, best_model)
        
        print(f"Loading original model: {best_model}")
        try:
            original_model, _ = load_model(best_model_path, SentimentModel, bert_tokenizer, device)
            models_info["Original Model"] = (original_model, SentimentModel)
        except Exception as e:
            print(f"Error loading original model: {e}")
            print("Skipping original model.")
      # Load enhanced model - only if enhanced_run is valid
    if enhanced_run and os.path.exists(enhanced_run):
        model_dir = os.path.join(enhanced_run, "checkpoints")
        best_models = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
        
        if best_models:
            best_model = sorted(best_models, key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), reverse=True)[0]
            best_model_path = os.path.join(model_dir, best_model)
            
            print(f"Loading enhanced model: {best_model}")
            try:
                enhanced_model, _ = load_model(best_model_path, EnhancedSentimentModel, bert_tokenizer, device)
                models_info["Enhanced Model"] = (enhanced_model, EnhancedSentimentModel)
            except Exception as e:
                print(f"Failed to load as EnhancedSentimentModel, trying BalancedSentimentModel: {e}")
                try:
                    enhanced_model, _ = load_model(best_model_path, BalancedSentimentModel, bert_tokenizer, device)
                    models_info["Enhanced Model"] = (enhanced_model, BalancedSentimentModel)
                except Exception as e2:
                    print(f"Error loading enhanced model: {e2}")
                    print("Skipping enhanced model.")
    
    # Fallback to manually trained model if available
    if "Enhanced Model" not in models_info and enhanced_run:
        manual_model_path = os.path.join(os.path.join(enhanced_run, "checkpoints"), "manual_trained_model.pt")
        if os.path.exists(manual_model_path):
            print("Loading manually trained model as fallback...")
            try:
                enhanced_model, _ = load_model(manual_model_path, BalancedSentimentModel, bert_tokenizer, device)
                models_info["Enhanced Model"] = (enhanced_model, BalancedSentimentModel)
            except Exception as e:
                print(f"Error loading manual model: {e}")
    
    # Compare models if we have at least two
    if len(models_info) >= 2:
        print("\nComparing models...")
        results = compare_models(models_info, test_loader, device)
        
        # Plot comparison metrics
        plot_comparison(results)
        
        # Plot confusion matrices
        plot_confusion_matrices(results)
        
        # Analyze errors
        print("\nAnalyzing error patterns across models...")
        error_analysis = analyze_errors(results, test_loader, bert_tokenizer)
        
        print(f"\nFound {len(error_analysis)} examples with errors.")
        print("Model comparison plots and error analysis saved.")
    else:
        print("Not enough models to compare.")

if __name__ == "__main__":
    main()

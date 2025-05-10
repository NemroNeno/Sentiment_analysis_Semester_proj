import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer
from models import SentimentModel
from dataset import load_data, prepare_tokenizer, SentimentDataset
from torch.utils.data import DataLoader
from utils import load_model

def analyze_dataset(data):
    """Analyze the dataset distribution"""
    # Count sentiment labels
    polarity_counts = Counter([item['polarity'] for item in data])
    labels_counts = Counter([item['label'] for item in data])
    
    print("\n=== Dataset Distribution ===")
    print("Polarity counts (-1, 0, 1):", polarity_counts)
    print("Label counts (0, 1, 2):", labels_counts)
    
    # Visualize distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Polarity plot
    sns.barplot(x=list(polarity_counts.keys()), 
                y=list(polarity_counts.values()), 
                palette="viridis", 
                ax=ax1)
    ax1.set_title("Distribution of Polarity Values")
    ax1.set_xlabel("Polarity (-1: Negative, 0: Neutral, 1: Positive)")
    ax1.set_ylabel("Count")
    
    # Label plot
    sns.barplot(x=list(labels_counts.keys()), 
                y=list(labels_counts.values()), 
                palette="viridis", 
                ax=ax2)
    ax2.set_title("Distribution of Sentiment Labels")
    ax2.set_xlabel("Label (0: Negative, 1: Neutral, 2: Positive)")
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("data_distribution.png")
    plt.close()
    
    return polarity_counts, labels_counts

def find_misclassified_examples(model, data_loader, tokenizer, device, num_examples=10):
    """Find examples where the model misclassifies, especially positive predicted as negative"""
    model.eval()
    misclassified = []
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            # Find misclassified examples
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    # Get the original text
                    decoded_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    
                    # Get probabilities
                    probs = torch.nn.functional.softmax(outputs[i], dim=0)
                    
                    misclassified.append({
                        'text': decoded_text,
                        'true_label': label.item(),
                        'true_sentiment': label_map[label.item()],
                        'predicted_label': pred.item(),
                        'predicted_sentiment': label_map[pred.item()],
                        'negative_prob': probs[0].item(),
                        'neutral_prob': probs[1].item(),
                        'positive_prob': probs[2].item()
                    })
            
            # If we have enough examples, break
            if len(misclassified) >= num_examples:
                break
    
    return misclassified[:num_examples]

def main():
    # Load data
    print("Loading data...")
    data = load_data(f"data\\final_dataset.json")
    
    # Analyze dataset distribution
    polarity_counts, label_counts = analyze_dataset(data)
    
    # Prepare tokenizer
    print("\nPreparing tokenizer...")
    bert_tokenizer, num_emoji_tokens = prepare_tokenizer(data)
    
    # Create dataset and dataloader
    test_dataset = SentimentDataset(data, bert_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find most recent run directory
    run_dirs = sorted([os.path.join("runs", d) for d in os.listdir("runs")], key=os.path.getmtime)
    
    if not run_dirs:
        print("No training runs found in the 'runs' directory.")
        return
    
    latest_run = run_dirs[-1]
    print(f"Analyzing model from latest run: {latest_run}")
    
    # Load best model
    model_dir = os.path.join(latest_run, "checkpoints")
    best_models = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
    
    if not best_models:
        print("No best model found in checkpoints directory.")
        return
    
    best_model = sorted(best_models, key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), reverse=True)[0]
    best_model_path = os.path.join(model_dir, best_model)
    
    print(f"Loading best model: {best_model}")
    model, checkpoint = load_model(best_model_path, SentimentModel, bert_tokenizer, device)
    
    # Find misclassified examples
    print("\nFinding misclassified examples...")
    misclassified = find_misclassified_examples(model, test_loader, bert_tokenizer, device, num_examples=20)
    
    # Display misclassified examples
    print("\n=== Misclassified Examples ===")
    
    # Count specific misclassifications
    positive_as_negative = 0
    negative_as_positive = 0
    
    for i, example in enumerate(misclassified):
        print(f"\nExample {i+1}:")
        print(f"Text: {example['text']}")
        print(f"True sentiment: {example['true_sentiment']}")
        print(f"Predicted sentiment: {example['predicted_sentiment']}")
        print(f"Probabilities: Negative: {example['negative_prob']:.4f}, " +
              f"Neutral: {example['neutral_prob']:.4f}, Positive: {example['positive_prob']:.4f}")
        
        # Count specific misclassifications
        if example['true_label'] == 2 and example['predicted_label'] == 0:  # Positive classified as Negative
            positive_as_negative += 1
        elif example['true_label'] == 0 and example['predicted_label'] == 2:  # Negative classified as Positive
            negative_as_positive += 1
    
    # Report on specific misclassification types
    print(f"\nPositive classified as Negative: {positive_as_negative}/{len(misclassified)}")
    print(f"Negative classified as Positive: {negative_as_positive}/{len(misclassified)}")
    
    # Save misclassified examples to a file
    with open("misclassified_examples.json", 'w') as f:
        json.dump(misclassified, f, indent=4)
    
    print("\nMisclassified examples saved to 'misclassified_examples.json'")
    print("Dataset distribution visualization saved to 'data_distribution.png'")

if __name__ == "__main__":
    main()

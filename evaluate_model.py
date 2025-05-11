import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer
from tqdm import tqdm

# Import local modules
from enhanced_models import BalancedSentimentModel
from dataset import load_data
from load_sentiment_model import create_model_from_checkpoint

def get_predictions(model, tokenizer, texts, labels, device='cpu', max_length=512, batch_size=32):
    """
    Get model predictions for a list of texts
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = labels
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_probabilities, all_labels

def plot_confusion_matrix(cm, class_names, output_path=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix with labels
    """
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    # Plot the confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Set labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, tokenizer, dataset_path, device='cpu', batch_size=32):
    """
    Evaluate the model on a dataset and generate confusion matrix
    """
    # Load data
    print(f"Loading dataset from {dataset_path}")
    data = load_data(dataset_path)
    
    # Extract texts and labels (assumes 'label' field in dataset)
    texts = []
    labels = []
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    for item in data:
        if 'text' in item and 'label' in item:
            texts.append(item['text'])
            labels.append(label_map.get(item['label'].lower(), 1))  # Default to neutral
    
    # Check if we have data
    if not texts:
        print("Error: No valid data found in the dataset")
        return
    
    print(f"Loaded {len(texts)} examples from dataset")
    
    # Get predictions
    predictions, probabilities, true_labels = get_predictions(
        model, tokenizer, texts, labels, device, batch_size=batch_size
    )
    
    # Create confusion matrix
    class_names = ['Negative', 'Neutral', 'Positive']
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate metrics
    report = classification_report(true_labels, predictions, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Calculate and print accuracy
    accuracy = (np.array(predictions) == np.array(true_labels)).mean()
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    # Output class-wise metrics
    print("\nClass-wise Performance:")
    for cls in class_names:
        print(f"{cls}:")
        print(f"  Precision: {report[cls.lower()]['precision']:.4f}")
        print(f"  Recall: {report[cls.lower()]['recall']:.4f}")
        print(f"  F1-score: {report[cls.lower()]['f1-score']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, 
        class_names, 
        output_path="confusion_matrix.png",
        title=f'Sentiment Analysis Confusion Matrix\nAccuracy: {accuracy:.4f}'
    )
    
    return cm, report

def main():
    parser = argparse.ArgumentParser(description='Evaluate sentiment model and generate confusion matrix')
    parser.add_argument('--model-path', type=str, default='runs/custom_model/kaggle2.pt', 
                        help='Path to the model weights file')
    parser.add_argument('--config-path', type=str, default='runs/custom_model/config.json', 
                        help='Path to the model config file')
    parser.add_argument('--dataset', type=str, default='data/final_dataset.json', 
                        help='Path to dataset file')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='.', 
                        help='Directory to save outputs')
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
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at {args.dataset}")
        return
    
    # Create model and load weights with correct architecture detection
    model, tokenizer = create_model_from_checkpoint(args.config_path, args.model_path, device)
    
    # Evaluate model
    cm, report = evaluate_model(
        model, tokenizer, args.dataset, device, batch_size=args.batch_size
    )
    
    # Save report to file
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    main()

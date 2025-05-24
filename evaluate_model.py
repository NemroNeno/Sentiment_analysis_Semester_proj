import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from transformers import BertTokenizer
from tqdm import tqdm

# Import local modules
from enhanced_models import BalancedSentimentModel
from dataset import load_data
from load_sentiment_model import create_model_from_checkpoint

# Example: Generate normalized confusion matrix
def plot_normalized_confusion_matrix(cm, class_names, accuracy, output_dir):
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.tight_layout()
    norm_cm_path = os.path.join(output_dir, "normalized_confusion_matrix.png")
    plt.savefig(norm_cm_path)
    print(f"Normalized confusion matrix saved to {norm_cm_path}")
    plt.close()  # Close plot to avoid memory issues

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

def evaluate_model(model, tokenizer, dataset_path, output_dir='.', device='cpu', batch_size=32, sample_ratio=0.1, max_samples=1000):
    """
    Evaluate the model on a dataset and generate confusion matrix
    
    Args:
        model: The sentiment model to evaluate
        tokenizer: The tokenizer for the model
        dataset_path: Path to the dataset file
        output_dir: Directory to save outputs
        device: Computing device (cpu/cuda)
        batch_size: Batch size for evaluation
        sample_ratio: Percentage of data to use (default: 0.1 = 10%)
        max_samples: Maximum number of samples to use (default: 1000)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    print(f"Loading dataset from {dataset_path}")
    
    try:
        data = load_data(dataset_path)
        print(f"Successfully loaded data with {len(data)} records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying to load directly as JSON...")
        try:
            # Try direct loading with explicit encoding
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            print(f"Successfully loaded data directly with {len(data)} records")
        except Exception as e2:
            print(f"All attempts to load dataset failed: {e2}")
            return None, None
            
    # Sample the data to use only a portion for evaluation
    total_samples = len(data)
    sample_size = min(int(total_samples * sample_ratio), max_samples)
    
    # Randomly sample the data
    import random
    random.seed(42)  # For reproducibility
    sampled_data = random.sample(data, sample_size)
    
    print(f"Sampled {sample_size} records ({sample_ratio*100:.1f}% of data, max {max_samples})")
    data = sampled_data
    
    # Extract texts and labels from dataset with flexible field handling
    texts = []
    labels = []
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    for item in data:
        # Find text field (could be 'text', 'sentence', 'content', etc.)
        text = None
        for field in ['text', 'sentence', 'content', 'message']:
            if field in item and item[field]:
                text = item[field]
                break
        
        # Find label field (could be 'label', 'sentiment', 'polarity', etc.)
        label = None
        if 'label' in item:
            label_value = item['label']
            if isinstance(label_value, int):
                # Numeric label (0, 1, 2)
                label = label_value if 0 <= label_value <= 2 else 1
            elif isinstance(label_value, str):
                # String label (e.g., 'negative', 'positive')
                label = label_map.get(label_value.lower(), 1)
        elif 'sentiment' in item:
            sent_value = item['sentiment']
            if isinstance(sent_value, int):
                label = sent_value if 0 <= sent_value <= 2 else 1
            elif isinstance(sent_value, str):
                label = label_map.get(sent_value.lower(), 1)
        elif 'polarity' in item:
            # Convert -1, 0, 1 to 0, 1, 2
            pol_value = item['polarity']
            if isinstance(pol_value, int) or isinstance(pol_value, float):
                label = int(pol_value) + 1 if -1 <= pol_value <= 1 else 1
        
        # Add to dataset if both text and label were found
        if text and label is not None:
            texts.append(text)
            labels.append(label)
    
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
      # Generate additional evaluation metrics
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
      # Plot confusion matrix
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm, 
        class_names, 
        output_path=output_path,
        title=f'Sentiment Analysis Confusion Matrix\nAccuracy: {accuracy:.4f}'
    )
    
    # Generate additional plots
    
    # 1. Save normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.tight_layout()
    norm_cm_path = os.path.join(args.output_dir, "normalized_confusion_matrix.png")
    plt.savefig(norm_cm_path)
    print(f"Normalized confusion matrix saved to {norm_cm_path}")
    
    # 2. Generate one-vs-rest ROC curves for each class
    plt.figure(figsize=(10, 8))
    
    # For storing AUC values
    roc_auc = {}
    
    # Convert true labels to one-hot encoding
    true_labels_onehot = np.zeros((len(true_labels), 3))
    for i, label in enumerate(true_labels):
        true_labels_onehot[i, label] = 1
    
    # Calculate ROC curve and ROC area for each class
    colors = ['blue', 'green', 'red']
    
    for i, cls in enumerate(class_names):
        # Get probabilities for this class
        class_probs = [prob[i] for prob in all_probabilities]
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(true_labels_onehot[:, i], class_probs)
        roc_auc[cls] = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{cls} (AUC = {roc_auc[cls]:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    plt.savefig(roc_path)
    print(f"ROC curves saved to {roc_path}")
    
    # 3. Save detailed metrics report as text file
    metrics_report = f"""SENTIMENT ANALYSIS MODEL EVALUATION REPORT
======================================
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {args.model_path}
Dataset: {args.dataset}
Samples evaluated: {len(true_labels)}

OVERALL METRICS:
---------------
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}
Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}

METRICS BY CLASS:
---------------"""

    for cls in class_names:
        metrics_report += f"""
{cls}:
  Precision: {report[cls.lower()]['precision']:.4f}
  Recall: {report[cls.lower()]['recall']:.4f}
  F1-Score: {report[cls.lower()]['f1-score']:.4f}
  Support: {report[cls.lower()]['support']}
  ROC AUC: {roc_auc.get(cls, 0.0):.4f}"""

    metrics_report += """

CONFUSION MATRIX:
---------------
"""
    metrics_report += f"             {' '.join([f'{cls:>10}' for cls in class_names])}\n"
    for i, cls in enumerate(class_names):
        metrics_report += f"{cls:10} {' '.join([f'{cm[i,j]:10d}' for j in range(len(class_names))])}\n"
    
    # Add normalized confusion matrix
    metrics_report += """
NORMALIZED CONFUSION MATRIX (row):
-------------------------------
"""
    metrics_report += f"             {' '.join([f'{cls:>10}' for cls in class_names])}\n"
    for i, cls in enumerate(class_names):
        metrics_report += f"{cls:10} {' '.join([f'{cm_normalized[i,j]:10.4f}' for j in range(len(class_names))])}\n"

    # Write full report to file
    report_txt_path = os.path.join(args.output_dir, "model_evaluation_report.txt")
    with open(report_txt_path, 'w') as f:
        f.write(metrics_report)
    
    print(f"Detailed evaluation report saved to {report_txt_path}")
    
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
        model, tokenizer, args.dataset, 
        output_dir=args.output_dir, 
        device=device, 
        batch_size=args.batch_size,
        sample_ratio=0.1,
        max_samples=1000
    )
    
    # Save report to file
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    main()

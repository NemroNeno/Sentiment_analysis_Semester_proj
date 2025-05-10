# Model Performance Improvement Guide

This document outlines the strategies implemented to improve the sentiment model's performance, specifically addressing the bias toward negative sentiment.

## Issue: Negative Sentiment Bias

Your sentiment model is showing a bias toward negative predictions, incorrectly classifying positive and neutral samples as negative. This issue can arise from several factors:

1. **Dataset Imbalance**: If the training data contains more negative examples than positive ones, the model may develop a bias toward the majority class.

2. **Feature Representation**: The model might be focusing on certain negative words or patterns that appear in both positive and negative contexts.

3. **Model Complexity**: The original model might not have enough capacity to capture subtle sentiment differences.

4. **Attention Mechanism**: The single-head attention may not be capturing the full context needed for accurate sentiment analysis.

## Solution Strategy

We've implemented multiple improvements to address these issues:

### 1. Enhanced Model Architecture (`enhanced_models.py`)

The enhanced model includes:

- **Multi-Head Attention**: Instead of single-head attention, the enhanced model uses multiple attention heads to capture different aspects of the input.
  ```python
  class MultiHeadAttention(nn.Module):
      def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
          # Multiple attention heads allow the model to focus on different parts of the input
  ```

- **Deeper LSTM Layers**: Multiple stacked LSTM layers to increase model capacity.
  ```python
  self.lstm = nn.LSTM(
      embedding_dim, 
      hidden_dim, 
      num_layers=num_lstm_layers,
      batch_first=True, 
      bidirectional=True,
      dropout=dropout if num_lstm_layers > 1 else 0
  )
  ```

- **Layer Normalization**: For more stable training.
  ```python
  self.norm1 = nn.LayerNorm(hidden_dim * 2)
  ```

- **Residual Connections**: To help with gradient flow in deeper networks.
  ```python
  output = self.layer_norm(output + residual)
  ```

- **Deeper Classification Layers**: A more expressive classifier network.
  ```python
  self.classifier = nn.Sequential(
      nn.Linear(hidden_dim * 2, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),  # GELU activation often performs better for NLP
      # ...more layers
  )
  ```

### 2. Class Weight Balancing (`BalancedSentimentModel` and `train_enhanced.py`)

To address dataset imbalance, we've added class weighting:

```python
class BalancedSentimentModel(EnhancedSentimentModel):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256, output_dim=3, 
                 num_lstm_layers=2, num_attention_heads=4, dropout=0.3, class_weights=None):
        # ...
        self.register_buffer('class_weights', 
                            torch.tensor(class_weights) if class_weights is not None 
                            else torch.ones(output_dim))
    
    def get_loss(self, outputs, labels):
        """Calculate weighted cross-entropy loss to address class imbalance"""
        return F.cross_entropy(outputs, labels, weight=self.class_weights)
```

The training script automatically calculates appropriate weights based on class frequencies:

```python
# Calculate class weights to address class imbalance
class_weights = calculate_class_weights(data)
```

### 3. Analysis Tools to Understand Bias

We've added diagnostic tools to help understand and address model bias:

- `analyze_bias.py`: Analyzes dataset distribution and finds misclassified examples
- `compare_models.py`: Compares performance between original and enhanced models

## How to Use the Improved Components

1. **Analyze Current Model Bias**:
   ```
   python analyze_bias.py
   ```
   This will help you understand specific examples where your model is failing.

2. **Train Enhanced Model**:
   ```
   python train_enhanced.py
   ```
   This will train a more complex model with class weighting.

3. **Compare Model Performance**:
   ```
   python compare_models.py
   ```
   After both models are trained, this will give you a side-by-side comparison.

## Expected Improvements

The enhanced model should show:
- Better accuracy and F1 scores overall
- More balanced performance across sentiment classes
- Reduced bias toward negative sentiment
- Better handling of context and subtle sentiment cues

## Fine-tuning Options

If the model is still underperforming on specific classes, you can:

1. **Adjust Class Weights**: Increase the weight for the positive class if it's still being misclassified.

2. **Augment Training Data**: Add more positive examples to balance the dataset.

3. **Model Parameters**: Try increasing `hidden_dim` or `num_attention_heads` for even more capacity.

4. **Learning Rate**: Use a smaller learning rate (1e-5 or 5e-6) for more stable training.

5. **Training Duration**: Train for more epochs to allow the model to converge.

# Sentiment Analysis Model

This project implements an enhanced sentiment analysis model using PyTorch, BERT embeddings, and LSTM-based architectures. The model classifies text as negative, neutral, or positive sentiment with improved handling for balanced datasets.

## Project Structure

- `enhanced_models.py`: Advanced model architectures with improved performance and support for varied configurations
- `load_sentiment_model.py`: Utilities for loading models with architecture detection
- `dataset.py`: Data loading and preprocessing utilities with robust encoding support
- `evaluate_model.py`: Comprehensive model evaluation script with metrics and visualizations
- `train_enhanced.py`: Script to train enhanced models with class balancing
- `train_utils.py`: Training and evaluation loop functions
- `utils.py`: General utility functions
- `sentiment_demo.py`: Interactive script to test model predictions
- `sentiment_app.py`: Streamlit web interface for interactive sentiment analysis
- `test_json_loading.py`: Utility to diagnose dataset loading issues
- `requirements.txt`: Dependencies for the project

## Key Features

- Sentiment classification into negative, neutral, and positive categories
- Uses BERT embeddings with configurable LSTM-based architectures
- Robust model architecture detection when loading checkpoints
- Multiple model architectures:
  - Basic: Attention-based LSTM architecture
  - Enhanced: Multi-head attention with deeper LSTMs
  - Balanced: Class weighting to handle dataset imbalance
- Comprehensive evaluation with:
  - Confusion matrix (regular and normalized)
  - Precision, recall, and F1-score reporting
  - ROC curves for each sentiment class
  - Detailed evaluation reports
- Support for dataset sampling during evaluation to handle large datasets
- Robust dataset loading with:
  - UTF-8 encoding with fallbacks
  - Support for different dataset formats and field names
  - Error handling for problematic characters
- Model checkpointing and best model saving
- TensorBoard integration for training visualization

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset in the expected format (see `data/final_dataset.json` for reference)

3. Training is handled by `train_enhanced.py` (see model configuration in the file)

## Model Evaluation

To evaluate a trained model on a dataset:

```bash
python evaluate_model.py --model-path "runs/custom_model/kaggle2.pt" --config-path "runs/custom_model/config.json" --dataset "data/final_dataset.json" --output-dir "evaluation_output"
```

This will:
- Load the model with automatic architecture detection
- Sample the dataset (10%, max 1000 samples by default)
- Generate confusion matrices and performance metrics
- Save detailed reports and visualizations to the output directory

## Interactive Testing

To interactively test the model with custom input:

```bash
python sentiment_demo.py --model-path "runs/custom_model/kaggle2.pt" --config-path "runs/custom_model/config.json"
```

## Interactive Web Interface

To use the Streamlit-based web interface for sentiment analysis:

```bash
streamlit run sentiment_app.py
```

This will launch a browser window with an interactive application that lets you:
- Analyze the sentiment of individual sentences
- Process multiple sentences in batch mode
- View sentiment results with confidence scores and probability distributions
- Navigate through results of multiple sentences one by one

The application includes a sidebar for configuring model paths and settings.

## Folder Structure

- `data/`: Contains the datasets used for training and evaluation
- `runs/`: Contains trained model checkpoints, configurations and TensorBoard logs
- `evaluation_output/`: Contains evaluation results and visualizations
- `model_backups/`: Contains backup copies of trained models

## Model Architecture

The sentiment model uses an enhanced architecture:
- BERT embeddings for rich semantic representations
- Configurable number of bidirectional LSTM layers
- Attention mechanism to focus on important parts of the text
- Fully connected layer for final classification
- Support for class balancing to handle dataset imbalance

## Technical Notes

- The model loading script automatically detects architecture parameters from checkpoints
- PyTorch 2.6 compatibility is ensured with proper handling of `weights_only` parameter
- Dataset loading handles encoding issues with fallbacks from UTF-8 to latin1
- The evaluation script supports flexible sampling to handle large datasets
- Report generation includes detailed performance metrics for each sentiment class

## Example Model Loading and Inference

```python
import torch
from transformers import BertTokenizer
from load_sentiment_model import create_model_from_checkpoint, predict_sentiment

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model with automatic architecture detection
config_path = "runs/custom_model/config.json"
model_path = "runs/custom_model/kaggle2.pt"
model, tokenizer = create_model_from_checkpoint(config_path, model_path, device)

# Make a prediction
text = "I really enjoyed this movie! 😍"
result = predict_sentiment(text, model, tokenizer, device)
print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}%)")
```

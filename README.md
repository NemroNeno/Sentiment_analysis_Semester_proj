# 🤖 Sentiment Analysis with Sarcasm and Emoji Awareness


## 🧠 Overview

In modern digital communication, sarcasm and emojis can distort textual sentiment, making traditional sentiment analysis models unreliable. This project presents a **lightweight hybrid deep learning model** combining **Bi-directional LSTMs and Multi-Head Self-Attention**, trained on a blended dataset from Kaggle IMDB and custom sarcasm-emoji rich sentences using the **Google Gemini API**.

> 🎯 Achieved **90.90% Accuracy** and **0.91 F1-Score** using a resource-efficient architecture.

---

## 🚀 Features

- ✅ Sarcasm-Aware Sentiment Analysis  
- ✅ Emoji-Aware Contextual Understanding  
- ✅ Lightweight Hybrid Deep Learning Model  
- ✅ Multiple model architectures:
  - ✅ Basic: Attention-based LSTM architecture
  - ✅ Enhanced: Multi-head attention with deeper LSTMs and residual connections
  - ✅ Balanced: Class weighting to handle dataset imbalance
- ✅ Custom Dataset Generation via Gemini API  
- ✅ BERT Tokenizer with Emoji Vocabulary Extension  
- ✅ Fully Integrated PyTorch Training Pipeline  
- ✅ TensorBoard integration for training visualization
- ✅ Model checkpointing and best model saving
- ✅ Class-Weighted Loss Handling  
- ✅ Comprehensive metrics tracking (accuracy, F1 score, confusion matrix)
- ✅ Deployed on Hugging Face 🤗

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

## Features

- Sentiment classification into negative, neutral, and positive categories
- Uses BERT embeddings with custom fine-tuning
- Multiple model architectures:
  - Basic: Attention-based LSTM architecture
  - Enhanced: Multi-head attention with deeper LSTMs and residual connections
  - Balanced: Class weighting to handle dataset imbalance
- Emoji support in text processing
- TensorBoard integration for training visualization
- Comprehensive metrics tracking (accuracy, F1 score, confusion matrix)
- Model checkpointing and best model saving
- Bias analysis and error detection
- Model comparison tools

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

The sentiment model architecture:
- Embedding layer initialized with BERT embeddings
- Bidirectional LSTM for sequence processing
- Attention mechanism to focus on important parts of the text
- Fully connected layer for final classification

## Results

After training, results are saved to the `runs` directory with:
- Model checkpoints (best model and latest model)
- Training configuration
- Performance metrics (accuracy, F1 score)
- Confusion matrix visualization
- Detailed classification report

## Using Trained Models

### Enhanced Models for Better Performance

If your model shows bias or poor performance, you can train an enhanced model:

```bash
python train_enhanced.py
```

This will train a more complex model with:
- Multi-head attention mechanism
- Multiple LSTM layers
- Class weighting to handle imbalanced data
- Deeper classification layers with residual connections

Your original model remains intact, and a new model will be saved separately.

### Analyze Model Bias

To understand why your model might be biased toward certain classes:

```bash
python analyze_bias.py
```

This will:
- Analyze your dataset distribution
- Find examples where the model makes errors
- Identify patterns in misclassifications
- Save visualizations of class distribution

### Compare Model Performance

To compare the original and enhanced models:

```bash
python compare_models.py
```

This will generate comparative metrics and visualizations to help you understand the improvements.

### Analyze Results Without Retraining

If you've already completed training and want to analyze the results:

```bash
python analyze_results.py
```

This will load your best model from the most recent training run and generate visualizations and reports.

### Make Predictions with Trained Model

To use your trained model for sentiment analysis on new text:

```bash
python predict_sentiment.py
```

This script loads the best model from your most recent training run and allows you to:
- See predictions on example sentences
- Enter your own text for sentiment analysis

### Model Loading and Inference Example

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

---
## 👥 Authors

- **Muhammad Nabeel** — [@NemroNeno](https://github.com/NemroNeno)
- **Umar Farooq** — [@Umar-Farooq-2112](https://github.com/Umar-Farooq-2112)

---

## 🔗 Resources & References

- [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [GitHub Project Repo](https://github.com/NemroNeno/Sentiment_analysis_Semester_proj.git)
- [Hugging Face Model Deployment](https://huggingface.co/mnabeel12/sentiment_analysis/tree/main)

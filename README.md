# Sentiment Analysis Model

This project implements a sentiment analysis model using PyTorch and BERT embeddings. The model classifies text as negative, neutral, or positive sentiment.

## Project Structure

- `train.py`: Main script to run the training process
- `models.py`: Neural network model definitions
- `dataset.py`: Data loading and preprocessing utilities
- `utils.py`: General utility functions for model initialization and visualization
- `train_utils.py`: Training and evaluation loop functions
- `requirements.txt`: Dependencies for the project

## Features

- Sentiment classification into negative, neutral, and positive categories
- Uses BERT embeddings with custom fine-tuning
- Attention-based LSTM architecture for sequence understanding
- Emoji support in text processing
- TensorBoard integration for training visualization
- Comprehensive metrics tracking (accuracy, F1 score, confusion matrix)
- Model checkpointing and best model saving

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset in the expected format (see `data/final_dataset.json` for reference)

3. Run training:
   ```bash
   python train.py
   ```

## Training Visualization

You can visualize the training process using TensorBoard:

```bash
tensorboard --logdir=runs
```

This will show training metrics, evaluation results, and confusion matrices.

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

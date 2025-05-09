import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import emoji
from sklearn.model_selection import train_test_split
import json
import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        label = item['label']

        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """Load data from JSON file and preprocess it for sentiment analysis"""
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert polarity to labels: -1 -> 0 (negative), 0 -> 1 (neutral), 1 -> 2 (positive)
    for item in data:
        item['label'] = item['polarity'] + 1
    
    return data

def prepare_tokenizer(data, pretrained_model="bert-base-uncased"):
    """Initialize tokenizer and add emoji tokens"""
    # Initialize BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # Extract all unique emojis from the dataset
    unique_emojis = set()
    for item in data:
        emojis = [char for char in item['sentence'] if char in emoji.EMOJI_DATA]
        unique_emojis.update(emojis)

    # Add emojis directly as tokens to the tokenizer
    bert_tokenizer.add_tokens(list(unique_emojis))
    
    return bert_tokenizer, len(unique_emojis)

def create_data_loaders(data, tokenizer, batch_size=16, test_size=0.2, random_state=42):
    """Create train and test data loaders"""
    # Split dataset
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_dataset = SentimentDataset(train_data, tokenizer)
    test_dataset = SentimentDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

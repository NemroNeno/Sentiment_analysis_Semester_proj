import torch
import torch.nn as nn

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_weights = self.softmax(attention_scores)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights

# Sentiment Analysis Model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256, output_dim=3, dropout=0.3):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        context_vector, attention_weights = self.attention(lstm_output)
        output = self.dropout(context_vector)
        output = self.fc(output)
        return output

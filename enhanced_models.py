import torch
import torch.nn as nn
import torch.nn.functional as F

# Enhanced Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Make sure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for query, key, value
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        residual = x
        
        # Project inputs to queries, keys, and values
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add head dims
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Weighted sum
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Final projection + residual + norm
        out = self.out(context)
        out = self.dropout(out)
        return self.layer_norm(out + residual), weights

# Feed-forward network for transformer-style processing
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

# Enhanced Sentiment Analysis Model with deeper architecture
class EnhancedSentimentModel(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=768, hidden_dim=256, output_dim=3,
        num_lstm_layers=4, num_attention_heads=8, dropout=0.3
    ):
        super(EnhancedSentimentModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        # Two stacked attention + feed-forward blocks
        self.attention1 = MultiHeadAttention(hidden_dim * 2, num_heads=num_attention_heads, dropout=dropout)
        self.ff1 = FeedForward(hidden_dim * 2, ff_dim=hidden_dim * 4, dropout=dropout)
        self.attention2 = MultiHeadAttention(hidden_dim * 2, num_heads=num_attention_heads, dropout=dropout)
        self.ff2 = FeedForward(hidden_dim * 2, ff_dim=hidden_dim * 4, dropout=dropout)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
        # Compatibility fc
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc.out_features = output_dim

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = self.norm1(lstm_out)
        # First attention block
        x, _ = self.attention1(x, attention_mask)
        x = self.ff1(x)
        # Second attention block
        x, _ = self.attention2(x, attention_mask)
        x = self.ff2(x)
        # Pooling
        mask_exp = attention_mask.unsqueeze(-1).expand(x.size())
        x = x * mask_exp
        summed = torch.sum(x, dim=1)
        lengths = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1e-9)
        pooled = summed / lengths
        # Classifier
        return self.classifier(pooled)

# Balanced model with class weights
class BalancedSentimentModel(EnhancedSentimentModel):
    def __init__(
        self, vocab_size, embedding_dim=768, hidden_dim=256, output_dim=3,
        num_lstm_layers=4, num_attention_heads=8, dropout=0.3, class_weights=None
    ):
        super(BalancedSentimentModel, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim,
            num_lstm_layers, num_attention_heads, dropout
        )
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weights = torch.ones(output_dim, dtype=torch.float32)
        self.register_buffer('class_weights', weights)

    def get_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels, weight=self.class_weights)

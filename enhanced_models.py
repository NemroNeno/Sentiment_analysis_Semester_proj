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
        batch_size = x.size(0)
        seq_length = x.size(1)
        residual = x
        
        # Project inputs to queries, keys, and values
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add head dimensions
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection with residual connection and layer normalization
        output = self.out(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

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
        x = F.gelu(x)  # GELU activation (better than ReLU for many NLP tasks)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)  # Residual connection
        return x

# Enhanced Sentiment Analysis Model with deeper architecture
class EnhancedSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256, output_dim=3, 
                 num_lstm_layers=2, num_attention_heads=4, dropout=0.3):
        super(EnhancedSentimentModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM with multiple layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_lstm_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Layer normalization after LSTM
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads=num_attention_heads, dropout=dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(hidden_dim * 2, hidden_dim * 4, dropout)
          # Classifier layers with residual connections
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
        
        # Add compatibility attribute for code that checks for model.fc
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # This is just for compatibility
        self.fc.out_features = output_dim  # Make sure the out_features attribute is accessible

    def forward(self, input_ids, attention_mask):
        # Embed input tokens
        embedded = self.embedding(input_ids)
        
        # Apply LSTM
        lstm_output, _ = self.lstm(embedded)
        lstm_output = self.norm1(lstm_output)  # Layer normalization
        
        # Apply multi-head self-attention with mask
        attention_output, _ = self.attention(lstm_output, attention_mask)
        
        # Apply feed-forward network
        output = self.feed_forward(attention_output)
        
        # Global pooling (using attention mask to avoid padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(output.size())
        masked_output = output * mask_expanded
        sum_embeddings = torch.sum(masked_output, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Apply classifier
        logits = self.classifier(pooled_output)
        
        return logits

# BalancedSentimentModel adds class weights to handle imbalanced data
class BalancedSentimentModel(EnhancedSentimentModel):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256, output_dim=3, 
                 num_lstm_layers=2, num_attention_heads=4, dropout=0.3, class_weights=None):
        super(BalancedSentimentModel, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, 
            num_lstm_layers, num_attention_heads, dropout
        )
          # Store class weights for optional weighted loss calculation
        # Ensure weights are float32/Float to match PyTorch's expected type
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weights = torch.ones(output_dim, dtype=torch.float32)
            
        self.register_buffer('class_weights', weights)
        
    def get_loss(self, outputs, labels):
        """Calculate weighted cross-entropy loss to address class imbalance"""
        return F.cross_entropy(outputs, labels, weight=self.class_weights)

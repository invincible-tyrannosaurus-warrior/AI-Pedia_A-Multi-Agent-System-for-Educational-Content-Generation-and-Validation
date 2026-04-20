import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
MAX_SEQ_LENGTH = 10
VOCAB_SIZE = 100
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism implementation"""
    
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Ensure embedding dimension is divisible by number of heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear layers for query, key, and value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project to query, key, and value
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out

class TransformerEncoderLayer(nn.Module):
    """Single layer of the transformer encoder"""
    
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # First sub-layer: Multi-head attention with residual connection and layer norm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Second sub-layer: Feed-forward network with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class TransformerEncoder(nn.Module):
    """Complete transformer encoder with multiple layers"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length):
        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, embed_dim)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final linear layer for output
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
    def _create_positional_encoding(self, max_length, embed_dim):
        """Create positional encoding matrix"""
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_length, embed_dim)
    
    def forward(self, x, mask=None):
        # Get sequence length
        seq_len = x.size(1)
        
        # Apply embedding
        x = self.embedding(x) * (self.embed_dim ** 0.5)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply each transformer layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

def generate_sample_data(batch_size, seq_length, vocab_size):
    """Generate sample data for training"""
    # Create random sequences
    data = torch.randint(0, vocab_size, (batch_size, seq_length))
    # Create target sequences (shifted by one position)
    targets = torch.cat([data[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long)], dim=1)
    return data, targets

def train_transformer():
    """Main training function for the transformer model"""
    
    # Initialize the model
    model = TransformerEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore index 0 (padding)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    losses = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Generate sample data
        inputs, targets = generate_sample_data(BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
        
        # Forward pass
        outputs = model(inputs)
        
        # Reshape for loss calculation
        outputs = outputs.reshape(-1, VOCAB_SIZE)
        targets = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    
    return model, losses

def visualize_training_losses(losses):
    """Visualize training losses over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_3a8bb282603540d79f5cbd832a7adb74/assets/transformer_training_loss.png')
    plt.close()

def test_model(model, test_input):
    """Test the model on a sample input"""
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        predicted = torch.argmax(output, dim=-1)
        return predicted

# Main execution
if __name__ == "__main__":
    # Train the transformer model
    trained_model, training_losses = train_transformer()
    
    # Visualize training progress
    visualize_training_losses(training_losses)
    
    # Test the model with a simple example
    test_input = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LENGTH))
    predictions = test_model(trained_model, test_input)
    
    print("\nModel Testing:")
    print(f"Input sequence: {test_input[0].tolist()}")
    print(f"Predicted sequence: {predictions[0].tolist()}")
    
    print("\nTransformer architecture successfully implemented!")
    print("Key components demonstrated:")
    print("- Multi-head attention mechanism")
    print("- Transformer encoder layers")
    print("- Positional encoding")
    print("- Training process with loss visualization")
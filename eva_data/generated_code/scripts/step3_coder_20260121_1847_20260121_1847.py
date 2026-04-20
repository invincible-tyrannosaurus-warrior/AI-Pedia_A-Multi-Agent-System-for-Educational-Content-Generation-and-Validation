import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import os

# Ensure the output directory exists
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformers.
    This allows the model to understand the order of tokens in a sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to store positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    Allows the model to focus on different parts of the input sequence simultaneously.
    """
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear transformations for query, key, and value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final linear transformation
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear transformations
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear transformation
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)
        
        return out

class FeedForward(nn.Module):
    """
    Implements feed-forward neural network layer.
    Applied after attention in each transformer block.
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    """
    Single transformer block consisting of:
    1. Multi-head attention
    2. Layer normalization
    3. Feed-forward network
    4. Another layer normalization
    """
    def __init__(self, d_model, n_heads, d_ff):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # First attention block with residual connection and layer norm
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward block with residual connection and layer norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class SimpleTransformer(nn.Module):
    """
    Complete simple transformer model for demonstration purposes.
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Final output layer
        output = self.fc_out(x)
        return output

def generate_sample_data():
    """
    Generate sample data for training the transformer.
    Creates sequences of numbers with a pattern.
    """
    # Create sequences of numbers from 0 to 9
    sequences = []
    for i in range(1000):
        # Create sequence of length 10 with numbers 0-9
        seq = list(range(10))
        # Shuffle the sequence randomly
        np.random.shuffle(seq)
        sequences.append(seq)
    
    # Convert to tensors
    data = torch.tensor(sequences, dtype=torch.long)
    return data

def train_transformer():
    """
    Train the transformer model on sample data.
    """
    # Model parameters
    vocab_size = 10  # Numbers 0-9
    d_model = 64     # Model dimension
    n_heads = 4      # Number of attention heads
    d_ff = 128       # Feed-forward dimension
    n_layers = 2     # Number of transformer layers
    max_len = 10     # Maximum sequence length
    
    # Create model
    model = SimpleTransformer(vocab_size, d_model, n_heads, d_ff, n_layers, max_len)
    
    # Generate sample data
    data = generate_sample_data()
    
    # Create dataloader
    dataset = TensorDataset(data, data)  # Input and target are the same
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    losses = []
    num_epochs = 50
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data_batch, target_batch) in enumerate(dataloader):
            # Forward pass
            output = model(data_batch)
            
            # Reshape for loss calculation
            output = output.reshape(-1, vocab_size)
            target = target_batch.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model, losses

def visualize_training_losses(losses):
    """
    Visualize the training losses over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transformer_training_losses.png'))
    plt.close()

def demonstrate_attention():
    """
    Demonstrate how attention works by visualizing attention weights.
    """
    # Create a simple example
    vocab_size = 10
    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 2
    max_len = 5
    
    # Create a simple model
    model = SimpleTransformer(vocab_size, d_model, n_heads, d_ff, n_layers, max_len)
    
    # Create a simple input sequence
    input_seq = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    
    # Get attention weights from first layer
    with torch.no_grad():
        # Forward pass to get intermediate outputs
        x = model.embedding(input_seq) * math.sqrt(d_model)
        x = model.pos_encoding(x)
        
        # Get attention weights from first layer
        attention_layer = model.layers[0].attention
        Q = attention_layer.W_q(x)
        K = attention_layer.W_k(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(attention_layer.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Plot attention weights
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights[0].numpy(), cmap='viridis', aspect='auto')
        plt.title('Attention Weights Visualization')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.colorbar(label='Attention Weight')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transformer_attention_weights.png'))
        plt.close()

def main():
    """
    Main function to run the transformer demonstration.
    """
    print("Starting Transformer Demonstration...")
    
    # Train the transformer
    print("Training transformer model...")
    model, losses = train_transformer()
    
    # Visualize training progress
    print("Generating training loss plot...")
    visualize_training_losses(losses)
    
    # Demonstrate attention mechanism
    print("Visualizing attention weights...")
    demonstrate_attention()
    
    print("Demonstration completed! Check the assets folder for results.")

if __name__ == "__main__":
    main()
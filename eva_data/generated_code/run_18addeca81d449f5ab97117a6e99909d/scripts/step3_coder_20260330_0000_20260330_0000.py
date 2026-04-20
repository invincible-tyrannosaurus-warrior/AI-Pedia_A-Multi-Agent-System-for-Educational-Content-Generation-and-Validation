import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create sample data for demonstration
def generate_sample_data():
    """Generate sample sequences for attention demonstration"""
    # Sequence length and embedding dimension
    seq_len = 10
    embed_dim = 64
    
    # Generate query, key, and value vectors
    queries = torch.randn(seq_len, embed_dim)
    keys = torch.randn(seq_len, embed_dim)
    values = torch.randn(seq_len, embed_dim)
    
    return queries, keys, values

# Implement scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention
    
    Args:
        Q: Query matrix of shape (seq_len, embed_dim)
        K: Key matrix of shape (seq_len, embed_dim)  
        V: Value matrix of shape (seq_len, embed_dim)
    
    Returns:
        attention_output: Output of attention mechanism
        attention_weights: Attention weights for visualization
    """
    # Compute attention scores using dot product
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale the scores by sqrt(d_k)
    d_k = Q.size(-1)
    scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # Apply attention weights to values
    attention_output = torch.matmul(attention_weights, V)
    
    return attention_output, attention_weights

# Implement multi-head attention
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, embed_dim=64, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final linear layer to combine heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        # Project inputs to get Q, K, V
        Q = self.q_proj(Q)
        K = self.k_proj(K)
        V = self.v_proj(V)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention for each head
        head_outputs = []
        for i in range(self.num_heads):
            head_Q = Q[:, i, :, :]
            head_K = K[:, i, :, :]
            head_V = V[:, i, :, :]
            
            # Compute attention for this head
            head_output, _ = scaled_dot_product_attention(head_Q, head_K, head_V)
            head_outputs.append(head_output)
        
        # Concatenate all heads
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Apply final projection
        output = self.out_proj(concatenated)
        
        return output

# Visualization function for attention weights
def visualize_attention_weights(attention_weights, title="Attention Weights"):
    """
    Visualize attention weights as a heatmap
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(attention_weights.detach().numpy(), cmap='viridis', aspect='auto')
    
    # Add labels
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_18addeca81d449f5ab97117a6e99909d/assets/attention_weights.png')
    plt.close()

# Main demonstration function
def main():
    print("Demonstrating Attention Mechanism")
    print("=" * 40)
    
    # Generate sample data
    print("1. Generating sample sequences...")
    queries, keys, values = generate_sample_data()
    print(f"Query shape: {queries.shape}")
    print(f"Key shape: {keys.shape}")
    print(f"Value shape: {values.shape}")
    
    # Demonstrate scaled dot-product attention
    print("\n2. Computing scaled dot-product attention...")
    attention_output, attention_weights = scaled_dot_product_attention(queries, keys, values)
    print(f"Attention output shape: {attention_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Show some attention weights
    print("\nSample attention weights:")
    print(attention_weights[:3, :3])
    
    # Visualize attention weights
    print("\n3. Visualizing attention weights...")
    visualize_attention_weights(attention_weights, "Scaled Dot-Product Attention Weights")
    
    # Demonstrate multi-head attention
    print("\n4. Testing multi-head attention...")
    # Create batch dimension for multi-head attention
    batch_queries = queries.unsqueeze(0)  # Shape: (1, seq_len, embed_dim)
    batch_keys = keys.unsqueeze(0)
    batch_values = values.unsqueeze(0)
    
    # Initialize multi-head attention
    multi_head_attn = MultiHeadAttention(embed_dim=64, num_heads=8)
    
    # Apply multi-head attention
    output = multi_head_attn(batch_queries, batch_keys, batch_values)
    print(f"Multi-head attention output shape: {output.shape}")
    
    # Compare outputs
    print("\n5. Comparing attention mechanisms...")
    print("Original attention output (first few elements):")
    print(attention_output[:3, :3])
    print("\nMulti-head attention output (first few elements):")
    print(output[0, :3, :3])
    
    # Show how attention can focus on different parts
    print("\n6. Demonstrating attention focusing on specific positions...")
    
    # Create a simple example where attention should focus on position 5
    simple_queries = torch.zeros(5, 64)
    simple_keys = torch.zeros(5, 64)
    simple_values = torch.zeros(5, 64)
    
    # Set one query to be very different from others (should attract attention)
    simple_queries[0] = torch.ones(64) * 10
    simple_keys[0] = torch.ones(64) * 10
    simple_values[0] = torch.ones(64) * 10
    
    # Set other keys to be different
    simple_keys[1] = torch.ones(64) * 5
    simple_values[1] = torch.ones(64) * 5
    
    # Compute attention
    simple_attention_output, simple_attention_weights = scaled_dot_product_attention(
        simple_queries, simple_keys, simple_values
    )
    
    print("Simple attention weights (focus on first position):")
    print(simple_attention_weights)
    
    # Visualize simple attention
    visualize_attention_weights(simple_attention_weights, "Simple Attention Focus Example")
    
    print("\nAttention mechanism demonstration completed!")

if __name__ == "__main__":
    main()
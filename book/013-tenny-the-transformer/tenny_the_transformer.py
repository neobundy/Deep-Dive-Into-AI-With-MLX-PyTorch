import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multihead_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Self-attention part of the encoder
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        # Feed-forward part of the encoder
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        # Layer normalization helps to stabilize the learning process
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Dropout added after the self-attention and feed-forward outputs
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block followed by a residual connection and layer normalization.
        src2 = self.norm1(src + self.dropout1(self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]))
        # Feed-forward block followed by another residual connection and layer normalization.
        src = self.norm2(src2 + self.dropout2(self.ff(src2)))
        return src

# TransformerEncoder aggregates multiple TransformerEncoderLayer layers together.
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        # Listing out multiple encoder layers defined above.
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        # Final layer normalization for the output of the last encoder layer.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        src = self.norm(src)
        return src

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix that is large enough for any possible sequence.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        # Use sinusoidal functions for positional encoding.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register pe as a persistent buffer that is not a parameter, but should be part of the module's state.
        self.register_buffer('pe', pe)

    # Adds the positional encoding to the input tensor and applies dropout.
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    # Model initialization for the full transformer model for sequence-to-sequence tasks.
    def __init__(self, ntoken, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        # Initialize the positional encoding module
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        # Embedding layer that maps token indices to embedding vectors
        self.encoder = nn.Embedding(ntoken, embed_dim)
        # The sequence of transformer encoder layers
        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        # Final linear layer that decodes the transformer output back to token space
        self.decoder = nn.Linear(embed_dim, ntoken)
        # Weight initialization routine
        self.init_weights()

    # Initializes weights of the transformer model with random values for training stability.
    def init_weights(self):
        initrange = 0.1  # Range for the uniform initializer
        self.encoder.weight.data.uniform_(-initrange, initrange)   # Encoder weights
        self.decoder.bias.data.zero_()  # Decoder bias
        self.decoder.weight.data.uniform_(-initrange, initrange)   # Decoder weights

    # Defines the forward pass of the entire transformer model.
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pass input ids through the embedding layer, scaled by the square root of the embedding dimension.
        src = self.encoder(src) * math.sqrt(embed_dim)
        # Apply positional encoding to the embeddings.
        src = self.pos_encoder(src)
        # Pass the positionally encoded embeddings through the transformer encoder.
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        # Decode the transformer encoder output to logit predictions for each token in the sequence.
        output = self.decoder(output)
        return output


# Example Usage
ntokens = 2000  # Vocabulary size
embed_dim = 512  # Embedding dimension
num_heads = 8    # Number of heads in multi-head attention
ff_dim = 2048    # Dimension of feed forward network
num_layers = 6   # Number of transformer encoder layers
dropout = 0.2    # Dropout rate

model = TransformerModel(ntokens, embed_dim, num_heads, ff_dim, num_layers, dropout)

# Dummy input for testing
src = torch.randint(0, ntokens, (35, 20))  # Example sequence
output = model(src)
print(output)

print(model.state_dict().keys())
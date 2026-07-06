import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads 
        
        # Combined projection for Q, K, V 
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Lower-triangular causal mask buffer 
        self.register_buffer("bias", torch.tril(torch.ones(context_length, context_length))
                                    .view(1, 1, context_length, context_length))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Context), d_model [cite: 18]
        
        # 1. Project to Q, K, V and split heads
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape to (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention [cite: 20]
        # (B, n_heads, T, d_k) x (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_k ** 0.5)) 
        
        # Apply causal mask: mask out future positions with -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1) 
        
        # 3. Multiply by V and concatenate heads back 
        y = att @ v # (B, n_heads, T, T) x (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Concat heads
        
        return self.c_proj(y)
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # d_ff = 4 * d_model 
        self.w_1 = nn.Linear(d_model, d_ff) 
        self.w_2 = nn.Linear(d_ff, d_model) 
        self.gelu = nn.GELU() 

    def forward(self, x):
        # FFN(x) = GELU(xW1 + b1)W2 + b2 [cite: 24]
        return self.w_2(self.gelu(self.w_1(x)))
class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads,max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model) 
        self.attn = CausalMultiHeadAttention(d_model, n_heads,max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model) 
        self.ffn = PositionWiseFFN(d_model,d_ff)

    def forward(self, x):
        # x = x + SubLayer(LN(x)) layout 
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_heads, n_layers, max_seq_len):
        super().__init__()
        self. max_seq_len =  max_seq_len
        
        # Foundational Embeddings 
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model) 
        
        # Sequential Transformer Blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_ff, n_heads, max_seq_len) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) 
        
        #  Weight-Tying 
        self.lm_head.weight = self.token_embedding.weight 
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len
        
        # Forward pass through embedding layers
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # Pass through Decoder layers
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        # Calculate logits 
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten tensors for cross-entropy evaluation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

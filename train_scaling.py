
import os
import json
import requests
import torch
import torch.optim as optim
from models import DecoderTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing natively inside notebook cell using device: {device}", flush=True)

# 1. Dataset Downloading & Preprocessing Pipeline
def load_dataset():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "input.txt"
    
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("Downloading Tiny Shakespeare corpus...", flush=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
            
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char2int = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char2int[c] for c in text], dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size

train_data, val_data, vocab_size = load_dataset()
print(f"Dataset prepared. Vocab Size: {vocab_size} | Train tokens: {len(train_data)}", flush=True)

# 2. Continuous Chunk Batching Pipeline
def get_batch(split, batch_size, context_length, data_fraction=1.0):
    if split == 'train':
        limit = int(len(train_data) * data_fraction)
        data_source = train_data[:limit]
    else:
        data_source = val_data
        
    ix = torch.randint(len(data_source) - context_length, (batch_size,))
    x = torch.stack([data_source[i : i + context_length] for i in ix])
    y = torch.stack([data_source[i + 1 : i + context_length + 1] for i in ix])
    
    return x.to(device), y.to(device)

def count_non_embedding_params(model):
    return sum(p.numel() for name, p in model.named_parameters()
               if 'embed' not in name and 'lm_head' not in name)

# 3. Standardized Core Training Execution Wrapper
def train_model(config, data_fraction=1.0, max_steps=3000, batch_size=32):
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        d_ff=config['d_ff'],   
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    N = count_non_embedding_params(model)
    print(f"\n[INIT] Scale: {config['name']} | Non-Embed Params (N): {N}", flush=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    best_val_loss = float('inf')
    model.train()
    
    for step in range(1, max_steps + 1):
        x_batch, y_batch = get_batch('train', batch_size, config['max_seq_len'], data_fraction)
        logits, loss = model(x_batch, y_batch)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 500 == 0:
            print(f"    Step {step:4d}/{max_steps} | Batch Train Loss: {loss.item():.4f}", flush=True)
            
        if step % 1000 == 0 or step == max_steps:
            model.eval()
            val_loss_accum = 0.0
            val_steps = 20
            with torch.no_grad():
                for _ in range(val_steps):
                    x_v, y_v = get_batch('val', batch_size, config['max_seq_len'])
                    _, v_loss = model(x_v, y_v)
                    val_loss_accum += v_loss.item()
                    
            current_val_loss = val_loss_accum / val_steps
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
            print(f" >> [EVAL] Step {step} | Current Best Val Loss: {best_val_loss:.4f}", flush=True)
            model.train()
                
    D = max_steps * batch_size * config['max_seq_len'] * data_fraction
    return N, D, best_val_loss

# 4. Main Sweeps Coordinator Loop
if __name__ == "__main__":
    model_profiles = {
        "Tiny":   {"name": "Tiny",   "n_layers": 2, "d_model": 64,  "d_ff": 256,  "max_seq_len": 128, "n_heads": 2},
        "Small":  {"name": "Small",  "n_layers": 4, "d_model": 128, "d_ff": 512,  "max_seq_len": 256, "n_heads": 4},
        "Medium": {"name": "Medium", "n_layers": 6, "d_model": 256, "d_ff": 1024, "max_seq_len": 256, "n_heads": 8}
    }

    results = {
        "parameter_sweep": [],
        "data_sweep": []
    }

    print("\n=== STARTING SWEEP 1: PARAMETER SCALING ===", flush=True)
    for scale_name in ["Tiny", "Small", "Medium"]:
        config = model_profiles[scale_name]
        N, _, best_loss = train_model(config, data_fraction=1.0)
        results["parameter_sweep"].append({"name": scale_name, "N": N, "loss": best_loss})
        
    print("\n=== STARTING SWEEP 2: DATA SCALING ===", flush=True)
    small_config = model_profiles["Small"]
    data_fractions = [0.10, 0.25, 0.50, 1.00]

    for fraction in data_fractions:
        if fraction == 1.00 and len(results["parameter_sweep"]) >= 2:
            small_run = results["parameter_sweep"][1]
            N = small_run["N"]
            D = 3000 * 32 * small_config['max_seq_len'] * 1.0
            best_loss = small_run["loss"]
            print(f"\n[CACHE] Using cached Small model metrics for {fraction*100}% data.", flush=True)
        else:
            N, D, best_loss = train_model(small_config, data_fraction=fraction)
            
        results["data_sweep"].append({"fraction": fraction, "D": D, "loss": best_loss})

    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n Sweeps Complete! Saved to sweep_results.json", flush=True)

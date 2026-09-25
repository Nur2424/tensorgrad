"""
examples/train_gpt.py
---------------------
Scale-up training run: tensorgrad GPTNano vs PyTorch baseline on Shakespeare

It trains both models for 2 000 steps, then writes:
 
    outputs/plots/training_loss.png          
    outputs/generated/shakespeare_generated.txt  
 
Usage
-----
    # from the project root:
    python examples/train_gpt.py
 
Output printed to stdout
------------------------
    Per-evaluation log lines (every 100 steps) for both models
    Final side-by-side val-loss comparison table
 
Notes
-----
* gc.collect() is called after every training step and every eval forward
  pass to break the circular reference inside backward closures
  (out._backward captures out, so Python's ref-counter alone cannot free
  the computation graph -- the cyclic GC must intervene explicitly).
* The PyTorch model uses the same architecture (pre-norm, combined c_attn)
  and the same init scheme (Kaiming uniform for Linear, Normal(0.02) for
  embeddings) so the loss curves are directly comparable.
* Both runs consume batches from the same numpy RNG (same seed) so they
  see the same sequence of training batches.
"""
 
import os, gc, sys, json, time, argparse, urllib.request
from datetime import datetime
import numpy as np
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
 
# examples/ is one level below the project root.
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn"
            "/master/data/tinyshakespeare/input.txt")
DATA_PATH = os.path.join(DATA_DIR, "shakespeare.txt")
 
# Make sure the tensorgrad package is importable when running from examples/
sys.path.insert(0, ROOT)
 
# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
 
HPARAMS = dict(
    # model architecture
    n_embd      = 256,
    n_head      = 4,
    n_layer     = 4,
    block_size  = 64,
    # training schedule
    batch_size    = 32,
    max_iters     = 5000,
    eval_interval = 100,
    eval_iters    = 50,
    # optimiser
    lr            = 3e-3,
    weight_decay  = 0.1,
    # data
    train_split   = 0.9,
    seed          = 1337,
)
 
 
# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
 
def load_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Shakespeare (~1 MB)...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH) as f:
        return f.read()
 
 
def build_codec(text):
    """Return (stoi, itos) for a character-level tokenizer."""
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    return stoi, itos
 
 
def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int64)
 
 
def get_batch(data, block_size, batch_size, rng):
    ix = rng.integers(0, len(data) - block_size, size=(batch_size,))
    x  = np.stack([data[i     : i + block_size    ] for i in ix])
    y  = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
 
 
def batch_stream(train_data, hp):
    """Yield (x, y) batches indefinitely; deterministic from hp['seed']."""
    rng = np.random.default_rng(hp['seed'])
    while True:
        yield get_batch(train_data, hp['block_size'], hp['batch_size'], rng)
 
 
# ---------------------------------------------------------------------------
# tensorgrad training
# ---------------------------------------------------------------------------
 
def train_tensorgrad(hp, train_data, val_data, batches):
    from tensorgrad.nn         import GPTNano
    from tensorgrad.optim      import AdamW
    from tensorgrad.functional import cross_entropy
 
    rng_eval = np.random.default_rng(hp['seed'] + 1)
 
    model = GPTNano(
        vocab_size = hp['vocab_size'],
        n_embd     = hp['n_embd'],
        n_head     = hp['n_head'],
        n_layer    = hp['n_layer'],
        block_size = hp['block_size'],
    )
    opt = AdamW(model.parameters(), lr=hp['lr'],
                weight_decay=hp['weight_decay'])
 
    print(f"  tensorgrad  |  {model.num_params():,} parameters")
 
    def estimate_loss():
        losses = {}
        for split, data in [('train', train_data), ('val', val_data)]:
            ls = []
            for _ in range(hp['eval_iters']):
                x, y   = get_batch(data, hp['block_size'], hp['batch_size'], rng_eval)
                logits = model(x)
                ls.append(float(cross_entropy(logits, y).data))
                # Break backward-closure circular refs between eval forward passes.
                gc.collect()
            losses[split] = float(np.mean(ls))
        return losses
 
    history = {'iter': [], 'train': [], 'val': []}
    t0 = time.time()
 
    for it in range(hp['max_iters'] + 1):
        if it % hp['eval_interval'] == 0:
            losses = estimate_loss()
            history['iter'].append(it)
            history['train'].append(losses['train'])
            history['val'].append(losses['val'])
            elapsed = time.time() - t0
            print(f"  iter {it:5d}  |  train {losses['train']:.4f}"
                  f"  |  val {losses['val']:.4f}  |  {elapsed:.1f}s")
 
        if it == hp['max_iters']:
            break
 
        # Cosine LR decay: lr_t = lr * 0.5 * (1 + cos(pi * t / T))
        lr_t = hp['lr'] * 0.5 * (1.0 + np.cos(np.pi * it / hp['max_iters']))
        opt.lr = lr_t
 
        opt.zero_grad()
        x, y   = next(batches)
        logits = model(x)
        loss   = cross_entropy(logits, y)
        loss.backward()
 
        # Gradient clipping (global norm <= 1.0)
        _params = list(model.parameters())
        total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in _params))
        if total_norm > 1.0:
            for p in _params:
                p.grad *= 1.0 / total_norm
 
        opt.step()
        # Explicit GC to free the computation graph between steps.
        gc.collect()
 
    # Save checkpoint so you can generate text later without retraining.
    ckpt_dir = os.path.join(OUT_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'tg_model.npz')
    np.savez(ckpt_path, **{f'p{i}': p.data for i, p in enumerate(model.parameters())})
    print(f"  checkpoint  ->  {ckpt_path}")
 
    return history, model
 
 
# ---------------------------------------------------------------------------
# PyTorch baseline  (pre-norm, same init as tensorgrad)
# ---------------------------------------------------------------------------
 
def train_torch(hp, train_data, val_data, batches):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
 
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"  pytorch     |  device: {device}")
    V      = hp['vocab_size']
    ne     = hp['n_embd']
    nh     = hp['n_head']
    dh     = ne // nh
    bs     = hp['block_size']
 
    # ---- module definitions ----
 
    class MHA(tnn.Module):
        def __init__(self):
            super().__init__()
            self.nh, self.dh = nh, dh
            self.c_attn = tnn.Linear(ne, 3 * ne)
            self.c_proj = tnn.Linear(ne, ne)
            k = ne ** -0.5
            tnn.init.uniform_(self.c_attn.weight, -k, k)
            tnn.init.zeros_(self.c_attn.bias)
            tnn.init.uniform_(self.c_proj.weight, -k, k)
            tnn.init.zeros_(self.c_proj.bias)
            mask = torch.triu(torch.ones(bs, bs, dtype=torch.bool), diagonal=1)
            self.register_buffer('mask', mask)
 
        def forward(self, x):
            B, T, C = x.shape
            qkv = self.c_attn(x)
            q = qkv[..., :ne      ].reshape(B, T, nh, dh).transpose(1, 2)
            k = qkv[..., ne:2*ne  ].reshape(B, T, nh, dh).transpose(1, 2)
            v = qkv[..., 2*ne:    ].reshape(B, T, nh, dh).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * (dh ** -0.5)
            attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
            attn = F.softmax(attn, dim=-1)
            y    = (attn @ v).transpose(1, 2).reshape(B, T, C)
            return self.c_proj(y)
 
    class FFN(tnn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = tnn.Linear(ne, 4 * ne)
            self.fc2 = tnn.Linear(4 * ne, ne)
            k1 = ne ** -0.5;      tnn.init.uniform_(self.fc1.weight, -k1, k1)
            tnn.init.zeros_(self.fc1.bias)
            k2 = (4 * ne) ** -0.5; tnn.init.uniform_(self.fc2.weight, -k2, k2)
            tnn.init.zeros_(self.fc2.bias)
 
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))
 
    class Block(tnn.Module):
        def __init__(self):
            super().__init__()
            self.ln1  = tnn.LayerNorm(ne)
            self.attn = MHA()
            self.ln2  = tnn.LayerNorm(ne)
            self.ff   = FFN()
 
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x
 
    class GPTPyTorch(tnn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = tnn.Embedding(V, ne)
            self.pos_emb   = tnn.Embedding(bs, ne)
            self.blocks    = tnn.ModuleList([Block() for _ in range(hp['n_layer'])])
            self.ln_f      = tnn.LayerNorm(ne)
            self.head      = tnn.Linear(ne, V, bias=False)
            tnn.init.normal_(self.token_emb.weight, std=0.02)
            tnn.init.normal_(self.pos_emb.weight,   std=0.02)
            tnn.init.uniform_(self.head.weight, -(ne ** -0.5), ne ** -0.5)
 
        def forward(self, idx):
            B, T = idx.shape
            pos  = torch.arange(T, device=device)
            x    = self.token_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))
 
    model = GPTPyTorch().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  pytorch     |  {n_params:,} parameters")
 
    opt      = torch.optim.AdamW(model.parameters(), lr=hp['lr'],
                                 weight_decay=hp['weight_decay'])
    rng_eval = np.random.default_rng(hp['seed'] + 1)
 
    def to_t(arr):
        return torch.from_numpy(arr).long().to(device)
 
    def estimate_loss():
        model.eval()
        losses = {}
        with torch.no_grad():
            for split, data in [('train', train_data), ('val', val_data)]:
                ls = []
                for _ in range(hp['eval_iters']):
                    x, y = get_batch(data, hp['block_size'], hp['batch_size'], rng_eval)
                    logits = model(to_t(x))
                    loss   = F.cross_entropy(logits.view(-1, V), to_t(y).view(-1))
                    ls.append(loss.item())
                losses[split] = float(np.mean(ls))
        model.train()
        return losses
 
    history = {'iter': [], 'train': [], 'val': []}
    t0 = time.time()
 
    for it in range(hp['max_iters'] + 1):
        if it % hp['eval_interval'] == 0:
            losses = estimate_loss()
            history['iter'].append(it)
            history['train'].append(losses['train'])
            history['val'].append(losses['val'])
            elapsed = time.time() - t0
            print(f"  iter {it:5d}  |  train {losses['train']:.4f}"
                  f"  |  val {losses['val']:.4f}  |  {elapsed:.1f}s")
 
        if it == hp['max_iters']:
            break
 
        # Cosine LR decay
        lr_t = hp['lr'] * 0.5 * (1.0 + np.cos(np.pi * it / hp['max_iters']))
        for pg in opt.param_groups:
            pg['lr'] = lr_t
 
        opt.zero_grad()
        x, y = next(batches)
        logits = model(to_t(x))
        loss   = F.cross_entropy(logits.view(-1, V), to_t(y).view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
 
    return history, model
 
 
# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
 
def save_plot(tg_history, pt_history):
    import matplotlib
    matplotlib.use('Agg')   # headless backend
    import matplotlib.pyplot as plt
 
    os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)
    path = os.path.join(OUT_DIR, 'plots', 'training_loss.png')
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    iters_tg = tg_history['iter']
    iters_pt = pt_history['iter']
 
    # Validation curves (solid)
    ax.plot(iters_tg, tg_history['val'],   color='#2563eb', lw=2,
            label='tensorgrad val')
    ax.plot(iters_pt, pt_history['val'],   color='#dc2626', lw=2,
            label='pytorch val',    linestyle='-')
 
    # Training curves (faint)
    ax.plot(iters_tg, tg_history['train'], color='#2563eb', lw=1,
            alpha=0.35, label='tensorgrad train')
    ax.plot(iters_pt, pt_history['train'], color='#dc2626', lw=1,
            alpha=0.35, label='pytorch train',    linestyle='--')
 
    ax.set_xlabel('iteration')
    ax.set_ylabel('cross-entropy loss (nats)')
    ax.set_title(
        f"tensorgrad vs PyTorch  "
        f"n_embd={HPARAMS['n_embd']}, n_head={HPARAMS['n_head']}, "
        f"n_layer={HPARAMS['n_layer']}, block={HPARAMS['block_size']}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved  ->  {path}")
 
 
def save_generated(model, stoi, itos, val_loss, n_params, n_chars=500):
    os.makedirs(os.path.join(OUT_DIR, 'generated'), exist_ok=True)
    path = os.path.join(OUT_DIR, 'generated', 'shakespeare_generated.txt')
 
    # Auto-regressive greedy generation
    prompt  = "ROMEO:"
    ctx     = np.array([[stoi.get(c, 0) for c in prompt]], dtype=np.int64)
    out_ids = list(ctx[0])
 
    for _ in range(n_chars):
        x      = ctx[:, -model.block_size:]
        logits = model(x)
        last   = logits.data[0, -1]          # (vocab_size,) logits at last position
        # Temperature sampling instead of greedy (avoids "the the the..." collapse)
        probs  = np.exp((last - last.max()) / 0.8)
        probs /= probs.sum()
        tok    = int(np.random.choice(len(probs), p=probs))
        out_ids.append(tok)
        ctx    = np.concatenate([ctx, [[tok]]], axis=1)
 
    generated = ''.join(itos[i] for i in out_ids)
 
    header = (
        "Engine:         tensorgrad\n"
        f"Val loss:       {val_loss:.4f}\n"
        f"Parameters:     {n_params:,}\n"
        f"Generated date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "-" * 60 + "\n\n"
    )
 
    with open(path, 'w') as f:
        f.write(header)
        f.write(generated)
        f.write('\n')
 
    print(f"Generated text  ->  {path}")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main():
    # ---- data ----
    text       = load_shakespeare()
    stoi, itos = build_codec(text)
    data       = encode(text, stoi)
    vocab_size = len(stoi)
    n          = int(HPARAMS['train_split'] * len(data))
    train_data = data[:n]
    val_data   = data[n:]
 
    hp = {**HPARAMS, 'vocab_size': vocab_size}
 
    print(f"\nShakespeare  |  {len(data):,} chars  |  vocab {vocab_size}")
    print(f"Split        |  {len(train_data):,} train  /  {len(val_data):,} val")
    print(
        f"Hyperparams  |  n_embd={hp['n_embd']}  n_head={hp['n_head']}"
        f"  n_layer={hp['n_layer']}  block={hp['block_size']}"
        f"  bs={hp['batch_size']}  iters={hp['max_iters']}\n"
    )
 
    # ---- tensorgrad ----
    print("=" * 55)
    print("tensorgrad")
    print("=" * 55)
    tg_batches = batch_stream(train_data, hp)
    tg_history, tg_model = train_tensorgrad(hp, train_data, val_data, tg_batches)
 
    # ---- pytorch ----
    print()
    print("=" * 55)
    print("pytorch baseline")
    print("=" * 55)
    pt_batches = batch_stream(train_data, hp)   # same seed => same batches
    pt_history, _ = train_torch(hp, train_data, val_data, pt_batches)
 
    # ---- comparison table ----
    tg_final = tg_history['val'][-1]
    pt_final = pt_history['val'][-1]
    diff     = abs(tg_final - pt_final)
    print()
    print("=" * 55)
    print("val loss comparison (final)")
    print("=" * 55)
    print(f"  tensorgrad  {tg_final:.4f}")
    print(f"  pytorch     {pt_final:.4f}")
    print(f"  |diff|      {diff:.4f}  {'OK' if diff < 0.30 else 'LARGE check gradients'}")
 
    # ---- save outputs ----
    save_plot(tg_history, pt_history)
    save_generated(
        tg_model, stoi, itos,
        val_loss = tg_final,
        n_params = tg_model.num_params(),
    )
 
    # Save raw histories for the notebook
    os.makedirs(os.path.join(OUT_DIR, 'json'), exist_ok=True)
    with open(os.path.join(OUT_DIR, 'json', 'tg_history.json'), 'w') as f:
        json.dump(tg_history, f, indent=2)
    with open(os.path.join(OUT_DIR, 'json', 'pt_history.json'), 'w') as f:
        json.dump(pt_history, f, indent=2)
 
    print("\nDone.")
 
 
if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap 
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='this is a demonstration program')
parser.add_argument('-batch_size', type=str, required=True, help='please provide a batch_size')

args = parser.parse_args()
print(f'batch size: {args.batch_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 128
batch_size = int(args.batch_size)
max_inters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2

print(device)

chars = ''
with open('openwebtext/vocab.txt', 'r',encoding = 'utf-8') as f:
    text=f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)

string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def get_random_chunk(split):
    filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            decode_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decode_block), dtype=torch.long)
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of seld attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_Embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # Validate token IDs and sequence length
        if index.min().item() < 0 or index.max().item() >= self.token_Embedding_table.num_embeddings:
            raise ValueError("Input token IDs are outside the vocabulary range")

        if T > block_size:
            raise ValueError("Input sequence exceeds block_size")

        tok_emb = self.token_Embedding_table(index)

        pos = torch.arange(T, device=index.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            if targets.min().item() < 0 or targets.max().item() >= logits.size(-1):
                raise ValueError("Target token IDs are outside the vocabulary range")

            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_new_tokens):
        self.eval()

        for _ in range(max_new_tokens):
            # Crop input to the maximum context length
            index_cond = index[:, -block_size:]

            logits, loss = self.forward(index_cond)

            # Get predictions for the last position
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            index_next = torch.multinomial(probs, num_samples=1)

            index = torch.cat((index, index_next), dim=1)

        return index

model = GPTLanguageModel(vocab_size)
print('loading model parameters....')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)

model = model.to(device)
print("Model loaded successfully!")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_inters):
    if iter % eval_iters ==0:
        losses = estimate_loss()
        print(f'step:{iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}')

    
    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')


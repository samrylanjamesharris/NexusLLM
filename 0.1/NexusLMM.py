# NexusLLM 0.1 (Unstable Build)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

block_size = 64
batch_size = 32
max_iters = 1514
eval_interval = 250
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 128
n_head = 4
n_layer = 2
dropout = 0.1

with open('0.1/NexusTraining.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos.get(int(i), '?') for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.95*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
model = MiniGPT().to(device)

if os.path.exists("0.1/NexusData.pth"):
    model.load_state_dict(torch.load("0.1/NexusData.pth", map_location=device))
    print("Loaded existing model. Skipping training.")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters+1):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % eval_interval == 0 or iter == max_iters:
            val_loss = model(*get_batch('val'))[1].item()
            print(f"Step {iter}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")
    torch.save(model.state_dict(), "0.1/NexusData.pth")
print("\nNexusLLM 0.1: An experimental large language model.")
print("Build: [Unstable]")
print("Lines of Training: [1500]")
print("")
history = ""
while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit"]:
            break
        history += "[USER]" + user_input + "\n[BOT]"
        input_ids = torch.tensor([encode(history)[-block_size:]], dtype=torch.long).to(device)
        output_ids = model.generate(input_ids, max_new_tokens=60)
        out = decode(output_ids[0].tolist())
        response = out[len(encode(history)):].split("[USER]")[0].strip()
        if len(response) == 0:
            response = out.split("[USER]")[-1].strip()
        print("LLM: ", end="")
        for char in response:
            print(char, end="", flush=True)
            time.sleep(0.02)
        print()
        history += "[BOT]" + response + "\n"
    except KeyboardInterrupt:
        print("\nExiting chat.")
        break
  

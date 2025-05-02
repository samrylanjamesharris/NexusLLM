# NexusLLM 0.0.1 (Experimental Build)
import numpy as np
import sys

embedding_dim = 128
seq_length = 64
num_heads = 4
head_dim = embedding_dim // num_heads
hidden_dim = 512
num_layers = 2

np.random.seed(42)

with open('NexusTraining.txt', 'r', encoding='utf-8') as f:
    text = f.read()

special_tokens = ['[USER]', '[BOT]', '\n']
chars = sorted(set(text + ''.join(special_tokens)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

def tokenize(s):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return ''.join([itos[t] for t in tokens])

params = {
    'token_embedding': np.random.randn(vocab_size, embedding_dim) * 0.01,
    'pos_embedding': np.random.randn(seq_length, embedding_dim) * 0.01,
}

for l in range(num_layers):
    params[f'q_{l}'] = np.random.randn(embedding_dim, embedding_dim) * 0.01
    params[f'k_{l}'] = np.random.randn(embedding_dim, embedding_dim) * 0.01
    params[f'v_{l}'] = np.random.randn(embedding_dim, embedding_dim) * 0.01
    params[f'o_{l}'] = np.random.randn(embedding_dim, embedding_dim) * 0.01
    params[f'fc1_{l}'] = np.random.randn(embedding_dim, hidden_dim) * 0.01
    params[f'fc2_{l}'] = np.random.randn(hidden_dim, embedding_dim) * 0.01
    params['lm_head'] = np.random.randn(embedding_dim, vocab_size) * 0.01

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

def softmax_1d(x):
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def reshape_heads(x, B, T):
    return x.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

def self_attention(x, layer):
    B, T, C = x.shape
    Q = x @ params[f'q_{layer}']
    K = x @ params[f'k_{layer}']
    V = x @ params[f'v_{layer}']
    Q = reshape_heads(Q, B, T)
    K = reshape_heads(K, B, T)
    V = reshape_heads(V, B, T)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    mask = np.tril(np.ones((T, T)))
    scores = np.where(mask == 0, -1e10, scores)
    weights = softmax(scores, axis=-1)
    out = weights @ V
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    return out @ params[f'o_{layer}']

def forward(x_tokens):
    B, T = x_tokens.shape
    x = params['token_embedding'][x_tokens] + params['pos_embedding'][:T]

    for l in range(num_layers):
        attn_out = self_attention(x, l)
        x = layer_norm(x + attn_out) 

        ff = np.maximum(0, x @ params[f'fc1_{l}']) @ params[f'fc2_{l}']
        x = layer_norm(x + ff) 

    logits = x @ params['lm_head']
    return logits

def sample(model_fn, input_tokens, num_tokens=30, temperature=1.0):
    output = list(input_tokens)

    for _ in range(num_tokens):
        x = np.array([output[-seq_length:]])
        logits = model_fn(x)[0, -1]
        logits = logits / temperature
        probs = softmax_1d(logits)
        next_token = np.random.choice(len(probs), p=probs)
        output.append(next_token)
    return output

history = ""
print("\nNexusLLM 0.0.1 - Build: Experimental - Training: 1000")
print("NexusLLM is a transformer-based language model. It generates responses based on limited training provied by the .txt flie.")
print("\nIf you would like to participate to make this AI better, go to the CONTRIBUTING file on GitHub to debug and look further into this.")
print("Cheers, Sam Harris.")
print("")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit"]:
            break
        history += f"[USER]{user_input}\n[BOT]"
        input_tokens = tokenize(history)[-seq_length:]
        output_tokens = sample(forward, input_tokens, num_tokens=60)
        response_tokens = output_tokens[len(input_tokens):]
        response_text = decode(response_tokens)
        response_text = response_text.split('[USER]')[0].split('\n')[0]
        print("Bot:", response_text.strip())
        history += response_text.strip() + '\n'

    except KeyboardInterrupt:
        print("\nExiting chat.")
        sys.exit()

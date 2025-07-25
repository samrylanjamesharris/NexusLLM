from ctransformers import AutoModelForCausalLM
import sys
import re

print("\nRecommended Models:")
print("1. TinyLlama 1.1B")
print("2. Llama 2 13B")
print("3. Mistral 2.5 7B - Recommened")

choice = ""
while choice not in ["1", "2", "3"]:
    choice = input("Select model (1-3): ").strip()
    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please enter 1, 2, or 3.")

if choice == "1":
    MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF"
    MODEL_FILE = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    MODEL_TYPE = "tinyllama"
elif choice == "2":
    MODEL_NAME = "TheBloke/Llama-2-13B-GGUF"
    MODEL_FILE = "llama-2-13b.Q4_K_M.gguf"
    MODEL_TYPE = "llama"
else:
    MODEL_NAME = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
    MODEL_FILE = "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
    MODEL_TYPE = "mistral"

GPU_LAYERS = 0
MAX_NEW_TOKENS = 256
CONTEXT_LENGTH = 4096
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.85
THREADS = 12
HISTORY_LIMIT = 3

print("\nLoading Model...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        model_file=MODEL_FILE,
        model_type=MODEL_TYPE,
        gpu_layers=GPU_LAYERS,
        context_length=CONTEXT_LENGTH,
        threads=THREADS,
        batch_size=128,
    )
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

conversation_history = []
INITIAL_PROMPT = (
    "You are NexusLLM, a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature. " 
    "If you don't know the answer to a question, please don't share false information."
    "Never include: <||>, </s>, <|nexusllm|>, <|user|>, <|assistant|>, or special tokens. "
    "Only respond with natural, human-like plain text. Keep it short and clear."
)

def sanitize_response(response):
    unwanted = [
        '</s>', '<|', '|>', '<||>', 
        '<|user|>', '<|nexusllm|>', '<|assistant|>',
        '<|system|>', r'\[INST\]', r'\[\/INST\]'
    ]
    for token in unwanted:
        response = response.replace(token, '')
    response = re.sub(r'<[^>]*>', '', response)
    response = re.sub(r'\[[^\]]*\]', '', response)
    return response.strip()

def trim_history(history):
    if len(history) > HISTORY_LIMIT:
        return history[-HISTORY_LIMIT:]
    return history

def build_prompt(history):
    prompt = f"<|system|>{INITIAL_PROMPT}</s>\n"
    for msg in history:
        role = "nexusllm" if msg["role"] == "bot" else "user"
        prompt += f"<|{role}|>{msg['content']}</s>\n"
    prompt += "<|nexusllm|>"
    return prompt

def get_bot_response(user_input, history):
    history.append({"role": "user", "content": user_input})
    history = trim_history(history)
    prompt = build_prompt(history)

    bot_response = ""
    print("NexusLLM: ", end="", flush=True)
    try:
        for token in model(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=1.3,
            stop=['</s>', '<|', '<|nexusllm|>', '<|user|>', '<|assistant|>]', '<|system|>']
            stream=True
        ):
            print(token, end="", flush=True)
            bot_response += token
        print()
    except Exception as e:
        print(f"\nGeneration error: {e}")
        return history, "There was an error with NexusLLM responding to your input..."

    bot_response = sanitize_response(bot_response)
    if any(bad_token in bot_response for bad_token in ['</s>', '<|']):
        bot_response = "[SYSTEM ERROR: Invalid tokens detected - response purged]"
        print("\nWARNING: Model attempted forbidden tokens!")

    history.append({"role": "bot", "content": bot_response})
    return history, bot_response

print("NexusLLM:")
print(f"Model: {MODEL_NAME}")
print(f"Threads: {THREADS} | Context: {CONTEXT_LENGTH} tokens | GPU Layers: {GPU_LAYERS}")
print("Type, 'exit' or 'quit' to close the AI chat. Type, 'clear' to reset the chat.")
if choice == "1":
    print("\nTinyLlama 1.1B is the smallest model supported on ctransformers and is VERY lightweight. This is ideal for fast responses and questions.\n")
if choice == "2":
    print("\nLlama 2 13B model is good for top-of-the-line responses. But, in order to use this model you need the recommended specifications.\n")
if choice == "3":
    print("\nMistral 2.5 7B model is good for performance and is newer than Llama 2. It is good for lower end systems and is lite on RAM.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("Memory cleared.")
        continue

    conversation_history, _ = get_bot_response(user_input, conversation_history)

def build_prompt(history):
    prompt = f"{INITIAL_PROMPT}\n\n"
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "bot":
            prompt += f"NexusLLM: {msg['content']}\n"
    prompt += "NexusLLM:"
    return prompt

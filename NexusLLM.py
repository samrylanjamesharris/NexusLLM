from ctransformers import AutoModelForCausalLM
import sys
import re

print("\nRecommended Models:")
print("1. Small: (Mistral-7B)")
print("2. Large: (Llama-2-13B)")
choice = input("Select model (1-2): ").strip()

if choice == "1":
    MODEL_NAME = "TheBloke/Mistral-7B-v0.1-GGUF"
    MODEL_FILE = "mistral-7b-v0.1.Q4_K_M.gguf"
    MODEL_TYPE = "mistral"
elif choice == "2":
    MODEL_NAME = "TheBloke/Llama-2-13B-Chat-GGUF"
    MODEL_FILE = "llama-2-13b-chat.Q4_K_M.gguf"
    MODEL_TYPE = "llama"
else:
    print("\nInvalid choice. Defaulting...")
    MODEL_NAME = "TheBloke/Mistral-7B-v0.1-GGUF"
    MODEL_FILE = "mistral-7b-v0.1.Q4_K_M.gguf"
    MODEL_TYPE = "mistral"

GPU_LAYERS = 0
MAX_NEW_TOKENS = 75
CONTEXT_LENGTH = 4096
TEMPERATURE = 0.5
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
INITIAL_PROMPT = "You are NexusLLM, a helpful and concise AI assistant. Your purpose is to provide accurate information and complete tasks as instructed. You always respond as NexusLLM, Never mimic the user. Never respond using, '<||>', '</s>', <|nexusllm|> or <|user|>. Keep your responses brief and to the point, always one sentence."

def sanitize_response(response):
    """Brute-force removal of all unwanted tokens and artifacts"""
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

def trim_history():
    global conversation_history
    if len(conversation_history) > HISTORY_LIMIT:
        conversation_history = conversation_history[-HISTORY_LIMIT:]

def get_bot_response(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    trim_history()
    prompt = f"<|system|>{INITIAL_PROMPT}</s>\n"
    for msg in conversation_history:
        role = "nexusllm" if msg["role"] == "bot" else "user"
        prompt += f"<|{role}|>{msg['content']}</s>\n"
    prompt += "<|nexusllm|>"
    
    try:
        bot_response = ""
        print("NexusLLM: ", end="", flush=True)
        for token in model(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=1.3,
            stop=['</s>', '<|'],
            stream=True
        ):
            print(token, end="", flush=True)
            bot_response += token
        print()
    except Exception as e:
        print(f"\nGeneration error: {e}")
        return "There was an error with NexusLLM responding to your input..."

    bot_response = sanitize_response(bot_response)
    if any(bad_token in bot_response for bad_token in ['</s>', '<|']):
        bot_response = "[SYSTEM ERROR: Invalid tokens detected - response purged]"
        print("\nWARNING: Model attempted forbidden tokens!")
    
    conversation_history.append({"role": "bot", "content": bot_response})
    return bot_response

print("NexusLLM - Multi-Model:")
print(f"Model: {MODEL_NAME}")
print(f"Threads: {THREADS} | Context: {CONTEXT_LENGTH} tokens | GPU Layers: {GPU_LAYERS}")
print("Type, 'exit' or 'quit' to close the AI chat. Type, 'clear' to reset the chat.")
if choice == "1":
    print("\nMistral can generate responses that give coherent, as well as giving somewhat correct information.\n")
else:
    print("\nFor top-of-the-line responses, Llama 2 is great for generating responses that are coherent all the time. Although, the time to respond may take a while since it is a larger model.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("Memory cleared.")
        continue

    bot_response = get_bot_response(user_input)

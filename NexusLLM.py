import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

print("\nRecommended Models:")
print("1. TinyLlama 1.1B")
print("2. Llama 2 13B")
print("3. Mistral 2.5 7B - Recommended")

choice = ""
while choice not in ["1", "2", "3"]:
    choice = input("Select model (1-3): ").strip()
    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please enter 1, 2, or 3.")

MODEL_MAP = {
    "1": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v0.3",
        "type": "tinyllama",
        "note": "\nTinyLlama 1.1B is the smallest model supported and is VERY lightweight. This is ideal for fast responses and questions.\n",
    },
    "2": {
        "name": "meta-llama/Llama-2-13b-hf",
        "type": "llama",
        "note": "\nLlama 2 13B model is good for top-of-the-line responses. But, in order to use this model you need the recommended specifications.\n",
    },
    "3": {
        "name": "TheBloke/Mistral-7B-v0.2-HF",
        "type": "mistral",
        "note": "\nMistral 2.5 7B model is good for performance and is newer than Llama 2. It is good for lower end systems and is lite on RAM.\n",
    },
}

selected = MODEL_MAP[choice]
MODEL_NAME = selected["name"]
MODEL_TYPE = selected["type"]
note = selected["note"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_NEW_TOKENS = 256
CONTEXT_LENGTH = 4096
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.85
HISTORY_LIMIT = 3

print("\nLoading Model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
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
    prompt = f"{INITIAL_PROMPT}\n\n"
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "bot":
            prompt += f"NexusLLM: {msg['content']}\n"
    prompt += "NexusLLM:"
    return prompt

def get_bot_response(user_input, history):
    history.append({"role": "user", "content": user_input})
    history = trim_history(history)
    prompt = build_prompt(history)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONTEXT_LENGTH).to(DEVICE)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    bot_response = ""

    print("NexusLLM: ", end="", flush=True)
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=1.3,
            streamer=streamer,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        bot_response = sanitize_response(generated_text)
        print(bot_response)
    except Exception as e:
        print(f"\nGeneration error: {e}")
        return history, "There was an error with NexusLLM responding to your input..."

    if any(bad_token in bot_response for bad_token in ['</s>', '<|']):
        bot_response = "[SYSTEM ERROR: Invalid tokens detected - response purged]"
        print("\nWARNING: Model attempted forbidden tokens!")

    history.append({"role": "bot", "content": bot_response})
    return history, bot_response

print("NexusLLM:")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE} | Context: {CONTEXT_LENGTH} tokens")
print("Type, 'exit' or 'quit' to close the AI chat. Type, 'clear' to reset the chat.")
print(note)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("Memory cleared.")
        continue

    conversation_history, _ = get_bot_response(user_input, conversation_history)

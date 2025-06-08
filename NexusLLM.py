from ctransformers import AutoModelForCausalLM
import time
import sys
import threading

MODEL_NAME = "MaziyarPanahi/WizardLM-2-7B-GGUF"
MODEL_FILE = "WizardLM-2-7B.Q4_K_M.gguf"
MODEL_TYPE = "llama"
GPU_LAYERS = 0
MAX_NEW_TOKENS = 75
CONTEXT_LENGTH = 4096
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
THREADS = 12
HISTORY_LIMIT = 3 

print(f"Loading model...")

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
INITIAL_PROMPT = "You are an AI model called NexusLLM, made to be a helpful and concise AI assistant. Respond conversationally in 1-2 short sentences."

def trim_history():
    global conversation_history
    if len(conversation_history) > HISTORY_LIMIT:
        conversation_history = conversation_history[-HISTORY_LIMIT:]

def streaming_typing_indicator(stop_event):
    animations = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    while not stop_event.is_set():
        for frame in animations:
            if stop_event.is_set():
                break
            sys.stdout.write(f"\rNexusLLM is responding... {frame} ")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")

def get_bot_response(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    trim_history()
    prompt = f"<|system|>{INITIAL_PROMPT}</s>\n"
    for msg in conversation_history:
        role = "assistant" if msg["role"] == "bot" else "user"
        prompt += f"<|{role}|>{msg['content']}</s>\n"
    prompt += "<|assistant|>"
    stop_event = threading.Event()
    typing_thread = threading.Thread(target=streaming_typing_indicator, args=(stop_event,))
    typing_thread.start()
    
    try:
        response = model(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=1.1,
            stream=False
        )
    except Exception as e:
        stop_event.set()
        typing_thread.join()
        print(f"\nGeneration error: {e}")
        return "There was an error with NexusLLM responding to your input..."
    
    stop_event.set()
    typing_thread.join()
    bot_response = response.strip().split('</s>')[0]
    bot_response = bot_response.replace("<|assistant|>", "").strip()
    conversation_history.append({"role": "bot", "content": bot_response})
    return bot_response

print("NexusLLM:")
print(f"Model: {MODEL_NAME}")
print(f"Threads: {THREADS} | Context: {CONTEXT_LENGTH} tokens")
print("")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("Memory cleared!")
        continue

    bot_response = get_bot_response(user_input)
    print(f"NexusLLM: {bot_response}")
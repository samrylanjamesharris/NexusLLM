from flask import Flask, render_template, request, jsonify
from ctransformers import AutoModelForCausalLM
import re
import os
import logging

app = Flask(__name__)
MODEL_CHOICE = os.getenv('MODEL_CHOICE', '1')
GPU_LAYERS = int(os.getenv('GPU_LAYERS', '0'))
THREADS = int(os.getenv('THREADS', '12'))

if MODEL_CHOICE == '1':
    MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    MODEL_FILE = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    MODEL_TYPE = "mistral"

MAX_NEW_TOKENS = 256
CONTEXT_LENGTH = 4096
TEMPERATURE = 0.5
TOP_K = 40
TOP_P = 0.85
HISTORY_LIMIT = 3
INITIAL_PROMPT = (
    "You are NexusLLM, a concise and helpful assistant. "
    "Never include: <||>, </s>, <|nexusllm|>, <|user|>, <|assistant|>, or special tokens. "
    "Only respond with natural, human-like plain text. Keep it short and clear."
)

print("\nLoading Model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        model_file=MODEL_FILE,
        model_type=MODEL_TYPE,
        gpu_layers=GPU_LAYERS,
        context_length=CONTEXT_LENGTH,
        threads=THREADS
    )
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Threads: {THREADS} | Context: {CONTEXT_LENGTH} tokens | GPU Layers: {GPU_LAYERS}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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
    return history[-HISTORY_LIMIT:] if len(history) > HISTORY_LIMIT else history

def build_prompt(history):
    prompt = f"<|system|>{INITIAL_PROMPT}</s>\n"
    for msg in history:
        role = "nexusllm" if msg["role"] == "assistant" else "user"
        prompt += f"<|{role}|>{msg['content']}</s>\n"
    prompt += "<|nexusllm|>"
    return prompt
@app.route('/')
def index():
    model_info = {
        '1': "Mistral 7B: Fast, Lightweight. "
    }
    return render_template('index.html', 
                           model_name=model_info[MODEL_CHOICE],
                           threads=THREADS,
                           context_length=CONTEXT_LENGTH)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    history = data.get('history', [])
    history.append({"role": "user", "content": user_input})
    history = trim_history(history)
    prompt = build_prompt(history)
    try:
        response = model(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=1.3,
            stop=['</s>', '<|']
        )
        bot_response = sanitize_response(response)
        
        if any(token in bot_response for token in ['</s>', '<|']):
            bot_response = "[SYSTEM: Invalid tokens detected - response purged]"
    except Exception as e:
        import logging
        logging.error("An error occureded during model 
processing", exc_info=True
        bot_response = "[SYSTEM: An internal error has
occurred. Please try again later.]"
    history.append({"role": "assistant", "content":
bot_response})
    
    return jsonify({
        'response': bot_response,
        'history': history
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR, format='%
(asctime)s - %(levelname)s - %(message)s')
    app.run(host='0.0.0.0', port=5000, debug=false)

# NexusLLM - Local AI / Offline:

A lightweight, privacy-focused AI powered entirely on your machine. Perfect for quick, AI interactions without relying on cloud APIs. You don't need any API key or anything to run this, once you follow the steps and run the script everything will automatically install on your system and can even run it offline without needing any internet connection. 

## Features:
- **Local & Private**: No data leaves your system, all of it stays there only. 
- **CPU and GPU Support**: This script is optimized for CPU (GPU optional for better preformance).

## Installation:
### Prerequisites
- Python 3.10+  
- 8GB+ RAM
- Visual Studio Code (Optional but is recommended to tweaking the script)

## Recommended Models:
- Small: Mistral-7B
- Large: Llama-2-13B

### Steps:
1. Clone the repo:  
   ```bash
   git clone https://github.com/samrylanjamesharris/NexusLLM.git
   cd NexusLLM
   
2. Install dependencies:

   ```bash
    pip install ctransformers

3. Download the model (automatically on first run, or manually):
- Auto-download: Just run the script. It'll automatically start downloading when you run the script.

That's it, you're basically done.
Just make sure to tweak the script so it matches your system, for example the threads its using and models. If you're not happy with the models you can switch them out by just going to Hugging Face and looking for models that are supported by ctransformers.

## Chat Example - Mistral-7B Model:
![aiexample](https://github.com/user-attachments/assets/9f636660-9884-472e-88c6-5f107bc1ee1c)


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
- Llama-2 7B - 4GB
- Llama-2 13B - 7GB
- Mistral 7B Instruct - 4GB

### Steps:
1. Clone the repo:  
   ```bash
   git clone https://github.com/samrylanjamesharris/NexusLLM.git
   cd NexusLLM
   
2. Install dependencies:

   ```bash
    pip install ctransformers

3. Download the model:
- Auto-download: Just run the script. It'll automatically start downloading when you run the script.

That's it, you're basically done.
Just make sure to tweak the script so it matches your system, for example the threads its using and models. If you're not happy with the models you can switch them out by just going to Hugging Face and looking for models that are supported by ctransformers.

### Web Steps:
Follow the same steps as before, Same process.
1. Clone the repo.
2. Install dependencies

   ```bash
   pip install ctransformers
   pip install flask
   
4. Run the script, automatically downloads model.
5. It runs the backbone of the entire thing and hosts the local script.
6. Go to your web browser and paste in
   ```bash
   http://localhost:5000

## Chat Example / Terminal - Mistral-7B Model:
![aiexample](https://github.com/user-attachments/assets/9f636660-9884-472e-88c6-5f107bc1ee1c)

## Chat Example / HTML Form - Mistral 7B Instruct Model:
![Screenshot_20250615_140930](https://github.com/user-attachments/assets/04a36b2a-2783-46f8-af23-b094aa17b24c)





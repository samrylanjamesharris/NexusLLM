# NexusLLM - Local AI / Offline:

A lightweight, privacy-focused AI powered entirely on your machine. Perfect for quick, AI interactions without relying on cloud APIs. You don't need any API key or anything to run this, once you follow the steps and run the script everything will automatically install on your system and can even run it offline without needing any internet connection. 

## Features:
- **Local**: No data leaves your system, all of it stays there only and runs completely on your syetem. 
- **CPU and GPU Support**: This script is optimized for CPU (GPU optional for better preformance).
- **Visual Studio Code**: You can use Viusal Studio Code to edit and the file in and out for more useability.

## Installation:
### Requirements Recommended:
- Python 3.11  
- 16GB RAM
- Visual Studio Code
- AMD Ryzen 5600G - (12 Threads - 6 Cores)

### Requirments Minimum:
- Python 3.10
- 4GB RAM
- Visual Studio Code
- AMD Athlon Silver 7120U (2 Cores - 2 Threads)


## Recommended Models:
- TinyLlama 1.1B / 668MB
- Llama 2 13B / 7GB
- Mistral 2.5 7B / 4GB - *Recomended*

#### If you have AI models that preform better than these, list them in the Issues tab in order to let us know.

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

## Chat Example / Terminal - Mistral 7B Model:
![aiexample](https://github.com/user-attachments/assets/9f636660-9884-472e-88c6-5f107bc1ee1c)

## Chat Example / HTML Form - Mistral Instruct 7B Model:
![Screenshot_20250615_140930](https://github.com/user-attachments/assets/04a36b2a-2783-46f8-af23-b094aa17b24c)

## Visual Guide For Visual Studio Code:
If you need a visual guide, click [here](https://youtu.be/c0v6siXTxxU) and follow the steps provided by Ethan.

<img width="854" height="480" alt="Mistral 2 5 7B" src="https://github.com/user-attachments/assets/4a3d1a8b-a6b8-469c-b646-f24066c7309f" />

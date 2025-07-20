# NexusLLM - Local AI / Ctransformers:

A lightweight, privacy-focused AI powered entirely on your machine. Perfect for quick, AI interactions without relying on cloud APIs. You don't need any API key or anything to run this, once you follow the steps and run the script everything will automatically install on your system and can even run it offline without needing any internet connection. 

## Features:
- **Local**: No data leaves your system, all of it stays there only and runs completely on your syetem. 
- **CPU and GPU Support**: This script is optimized for CPU (GPU optional for better preformance).
- **Visual Studio Code**: You can use Viusal Studio Code to edit and the file in and out for more useability.

## Installation:
### Requirements - Recommended:
- Python 3.11  
- 16GB RAM
- Visual Studio Code
- AMD Ryzen 5600G - (12 Threads - 6 Cores)

### Requirments - Minimum:
- Python 3.10
- 8GB RAM
- Visual Studio Code
- AMD Athlon Silver 7120U - (2 Cores - 2 Threads)


## Recommended Models:
- TinyLlama 1.1B / 668MB - *Smallest Model*
- Llama 2 13B / 7GB - *Biggest Model*
- Mistral 2.5 7B / 4GB - *Recomended Model*

#### If you have AI models that preform better than these, list them in the Issues tab in order to let us know.

## Steps:
1. Clone the repo:  
   ```bash
   git clone https://github.com/samrylanjamesharris/NexusLLM.git
   cd NexusLLM
   
2. Install dependencies:
   ```bash
    pip install ctransformers

3. Download the model:
- Auto-download: Just run the script. Pick a model out of the selection, it'll automatically start downloading when you select one.

That's it, you're basically done.
Make sure to tweak the script so it matches your system or preferences, it could be switching out the model or setting the temperture on the model. 
If you're not happy with the models already in the script, you can switch them out by just going to Hugging Face and looking for models that are supported by ctransformers. I use TheBloke's models which work 100% of the time, of course, you can use other ones if you want to.

## Chat Example / Terminal - Mistral 7B Model:
![aiexample](https://github.com/user-attachments/assets/9f636660-9884-472e-88c6-5f107bc1ee1c)

## Guide For Visual Studio Code:
If you need a visual guide, click [here](https://youtu.be/c0v6siXTxxU) and follow the steps provided in the video by Ethan.

<img width="854" height="480" alt="Mistral 2 5 7B" src="https://github.com/user-attachments/assets/4a3d1a8b-a6b8-469c-b646-f24066c7309f" />

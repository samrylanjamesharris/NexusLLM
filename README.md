# NexusLLM - [NexGen Community]:

**NexusLLM** is a experimental built large language model using NanoGPT. It is designed to be simple, debuggable, and open to all contributors.

---

> *NexGen* Community - *Nex*us / *Gen*eration - A connection between the community.

---
## Next Update: **NanoGPT**
The next 0.3 updatte will upgrade from `MiniGPT` to `NanoGPT`. You might be asking why we did this switch:

1. Unlike MiniGPT, NanoGPT can be used to train on large corpora. Such as, OpenWebText.
2. It has more support on GitHub and other platforms.
3. It is more scaleable from small to GPT-2 size and above.

---

# Installation ⬇️:
This will show you how to install *NexusLLM 0.2*, make sure you have the latest Python version installed:

1. *Download NexusLMM: Download the latest version of **NexusLMM 0.X** *(X stands for the latest model number.)* and put it wherever, just make sure inside the python file that it's using the right directory.*
   
2. *Install Packages: Install these listed packages inside the terminal:*
```pip install torch ```,
```pip install minigpt```,
```pip install numpy```.

4. *Run it: That's all, just make sure to run the script from the project root and it should be good.*

***IF THERE ARE ANY ERRORS OR INCORRECT INFORMATION INSIDE THIS INSTALLTION, MAKE SURE TO LET ME KNOW IN THE ISSUES TAB ON GITHUB AND I'LL ADDRESS THEM AS SOON AS POSSIBLE.***

---

# Contributing 🤝:
Do **you** want to help us make NexusLLM bigger and smarter?
If you're interested, go read the [CONTRIBUTING](https://github.com/samrylanjamesharris/NexusLLM/blob/main/CONTRIBUTING.md) flie to help debug and code!

---

# Features ⚒️:
• *Multi-head self-attention:* **Applying attention with a causal mask and merging the heads back**

• *Simple transformer:* **It has a standard 2-layer-MLP with layer normalization with a feedforward block.**

• *Token and position embeddings:* **It has positional embeddings which is essential for all transformations to have**

• *Sample with temperature control:* **It has a function which you can scale the temperature to whatever you want.**

• *NanoGPT:* **This is a small-scale GPT-like model that can be customizable and lightweight. This will also allow it to read off the training data provided.**

---

# Plans 🗓️:
*Here are our plans so far to make this a better experience moving on in the furture!*
| Version | Focus                        | 	Status |
|---------|------------------------------|----------|
| Alpha   | Alpha Build / 1000 Lines of Training | Outdated |
| 0.1     | Unstable Build / 1500 Lines of Training / 16,316 Tokens / MiniGPT | Outdated |
| 0.2     | Unstable Build / 5000 Lines of Training / 43,981 Tokens | Current Build |
| 0.3     | Unstable Build / 10,000 Lines of Training / 70,000 Tokens / NanoGPT | Planned |
| 0.4     | Stable Build / 20,000 Lines of Training / 150,000 Tokens   | Planned |
| 0.5     | Finished Build / 50,000 Lines of Training / 380,000 Tokens | Planned |

---

# Model Comparison 📊: 
### Alpha Model:
![Screenshot 2025-05-02 8 37 40 AM](https://github.com/user-attachments/assets/235704e9-147d-4c4e-a116-184c861e02ba)

**The alpha model generates a random string of characters and symbols, this was built from scratch using NumPy and limited data. Even if you gave it all of the information, it would still come out the same since this is toy trasnformer and not an LLM.**


### 0.1 Model - [16,316 Tokens]:
![Screenshot 2025-05-02 11 44 59 PM](https://github.com/user-attachments/assets/40ce2ea1-4709-41e6-8af4-2d8379723f88)

**Using MiniGPT and more information data, NexusLLM 0.1 makes an improvement. While the joke still isn't funny, it now understands English. It indicates that the 0.1 model is starting to learn patterns and sequences of language.**

### 0.2 Model - [43,981 Tokens]:
![0 2](https://github.com/user-attachments/assets/e6b0abfa-c39d-4d1d-a938-e1089191eafa)

**0.2 model uses 5000 lines of information, compared to the 1000 used from the 0.1 model. We are able to see that the AI is now able to finish senctences and say english words properly, (sometimes...) Overall, a slight improvement.**

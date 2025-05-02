# NexusLLM - [NexGen Community]:

**NexusLLM** is a experimental built large language model using MiniGPT. It is designed to be simple, debuggable, and open to all contributors.

---

***[ANNOUNCEMENT: NexusLLM 0.1 will be released at 3:00 PM NZST (UTC+12). This will include the new framework, Torch with MiniGPT included.]***

> *NexGen* Community - *Nex*us / *Gen*eration - A connection between the community.

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

• *MiniGPT:* **This will allow NexusLLM to read off the training data provied.**

---

# Plans 🗓️:
*Here are our plans so far to make this a better experience moving on in the furture!*
| Version | Focus                        | 	Status |
|---------|------------------------------|----------|
| 0.0.1   | Experimental Build / 1000 Lines of Training | Outdated |
| 0.1     | Unstable Build / 1500 Lines of Training + MiniGPT | Current Build |
| 0.2     | Unstable Build / 5000 Lines of Training | Planned |
| 0.3     | Unstable Build / Expand upon Python | Planned |
| 0.4     | Stable Build / 10,000 Lines of Training | Planned |
| 0.5     | Stable Build / Community fine-tunes | Planned |
| 1.0     | Finished Build / 50,000 Lines of Training | Planned for Late 2025 / Late 2026 |

---
# Model Comparison 📊: 
### 0.0.1 Model - [1.0 Temperature]:
![Screenshot 2025-05-02 8 37 40 AM](https://github.com/user-attachments/assets/235704e9-147d-4c4e-a116-184c861e02ba)

**0.0.1 generates a random string of characters and symbols, this was built from scratch using NumPy and limited data. Even if you gave it all of the information, it would still come out the same since this is toy trasnformer and not LLM.**


### 0.1 Model - [0.7 Temperature]:
![Screenshot 2025-05-02 11 44 59 PM](https://github.com/user-attachments/assets/40ce2ea1-4709-41e6-8af4-2d8379723f88)

**Using MiniGPT and more information data, NexusLLM 0.1 makes an improvement. While the joke still isn't funny, it now understands English. It indicates that the 0.1 model is starting to learn patterns and sequences of language.**


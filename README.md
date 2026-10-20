# GPT From Scratch (PyTorch)

A small GPT-style large language model built from scratch in Python and PyTorch. The project covers the full pipeline: data handling, tokenization, a bigram baseline, self-attention, a transformer-based GPT, training on OpenWebText, saving the model, and chatting with it from a script.

> **Credit:** This project follows the freeCodeCamp tutorial [Create a Large Language Model from Scratch with Python](https://youtu.be/UU1WVnMk4E8), developed by [Elliot Arledge](https://x.com/elliotarledge). All credit for the original course and approach goes to them.

## Project Structure

```
.
├── llm.ipynb            # Step-by-step notebook: data, tokenizer, bigram model, PyTorch basics
├── gpt-v1.ipynb         # Notebook version of the GPT (transformer) model
├── training.py          # Script to train the GPT model on OpenWebText
├── chatbox.py           # Script to load the trained model and chat / complete prompts
├── wizard_of_oz.txt     # Small text dataset used for early experiments
└── README.md
```

**Not included in the repo** (too large or machine-specific, see `.gitignore`):

- `openwebtext/` - the large training corpus, download it separately
- `model-01.pkl` - trained model weights, generated when you run `training.py`
- `cuda/` - local GPU setup files

## Features

- Character-level tokenizer (encode / decode)
- Train and validation split with random batch generation (`get_batch`)
- Bigram language model as a baseline
- Transformer components built by hand:
  - Token and positional embeddings
  - Scaled dot-product self-attention (scaled by 1/sqrt(d_k))
  - Multi-head attention
  - Feed-forward network
  - Residual connections and layer normalization
- `GPTLanguageModel` built from stacked transformer blocks
- Training loop with optimizer, loss reporting, and train/eval modes
- GPU (CUDA) training with CPU fallback
- Model saving and loading with pickle (weights are saved to `model-01.pkl` after training)
- Command-line argument parsing for hyperparameters
- `chatbox.py` for prompt completion with the trained model

## Tech Stack

- Python 3
- PyTorch
- Jupyter Notebook
- NVIDIA CUDA

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch numpy jupyter matplotlib
```

For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Get the data

- `wizard_of_oz.txt` is included and is good for quick experiments.
- For full training, download the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) corpus and place it in the `openwebtext/` folder.

## Usage

### Explore the notebooks

```bash
jupyter notebook
```

Open `llm.ipynb` first, then `gpt-v1.ipynb`.

### Train the model

```bash
python training.py -batch_size 32
```

This trains the GPT model and saves the weights to `model-01.pkl` in the project folder. The trained model is not included in this repo, so you need to run training yourself first.

### Chat with the trained model

Once `model-01.pkl` exists (after training), run:

```bash
python chatbox.py -batch_size 32
```

Type a prompt and the model will complete it.

> Adjust the arguments above to match the ones your scripts accept.

## What I Learned

**Data and text processing**
- How raw text is turned into numbers using a character-level tokenizer
- Splitting data into train and validation sets, and building input/target batches
- Handling a large dataset (OpenWebText) with a custom data loader instead of loading everything into memory

**PyTorch and deep learning fundamentals**
- Working with tensors, embeddings, matrix multiplication and dot products
- Building models with `nn.Module`, plus gradient descent, optimizers and `zero_grad`
- Activation functions (ReLU, Sigmoid, Tanh) and normalization
- Reading logits, reshaping tensors, and computing loss

**Transformers and GPT**
- How self-attention works and why scores are scaled by 1/sqrt(d_k)
- Multi-head attention, feed-forward layers and transformer blocks
- Positional encoding and how a GPT differs from the original encoder-decoder transformer
- Putting it together into a working `GPTLanguageModel`

**Training and engineering**
- Switching between CPU and GPU (CUDA) and monitoring GPU memory
- Using train vs. eval mode and reporting loss correctly
- Saving and loading trained models with pickle
- Moving code from notebooks into reusable scripts with command-line arguments
- Building a simple prompt-completion interface (`chatbox.py`)
- Debugging real errors along the way (shapes, devices, memory)
- The difference between pretraining and fine-tuning

## Future Improvements

- Replace the character-level tokenizer with BPE
- Train for longer on more data for better text quality
- Fine-tune on a custom dataset
- Add a simple web interface for the chatbox

## Acknowledgements

- [freeCodeCamp.org](https://www.freecodecamp.org) and [Elliot Arledge](https://x.com/elliotarledge) for the tutorial
- The PyTorch community

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

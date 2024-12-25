# RWKV Language Model - Eagle 7B Part III

## Training Preparations - `demo-training-prepare.sh'

The `demo-training-prepare.sh` script is a shell script designed to set up and initiate training for the RWKV Language Model. It performs several steps to prepare the environment, download the dataset, and configure training parameters before launching the training process.

```bash
#!/bin/bash

# Create data directory

mkdir -p data

# Download minipile (1498226207 tokens, around 3GB)

wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin

# Generate initial model (L12-D768 = 169M)

BASE_NAME="model/0.1-1"
N_LAYER="12"
N_EMBD="768"

# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case)
# use https://www.dcode.fr/prime-numbers-search

python train.py --wandb "" --proj_dir $BASE_NAME \
 --data_file "data/minipile" --data_type "binidx" --vocab_size 65536 \
 --ctx_len 512 --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 --enable_progress_bar False --ds_bucket_mb 200
```

### 1. Create Data Directory

- The script starts by creating a directory named `data` to store the dataset and potentially other data-related files.

### 2. Download Dataset

- It downloads the `minipile` dataset, which is a tokenized version of a text corpus, from Hugging Face's datasets repository. This dataset includes both index (`.idx`) and binary (`.bin`) files, which are used by the RWKV model for training. The dataset is about 3GB in size, containing approximately 1,498,226,207 tokens.

### 3. Generate Initial Model Configuration

- The script sets up variables for the base name of the model (`BASE_NAME`), the number of layers (`N_LAYER`), and the embedding size (`N_EMBD`). These variables are used to specify the model's architecture, in this case, 12 layers with an embedding size of 768, totaling around 169 million parameters.

### 4. Compute `magic_prime`

- A comment in the script explains that `magic_prime` should be the largest prime number of the form `3n+2` that is smaller than the ratio of the dataset length to the context length minus one. It's a specific parameter that might be used for dataset partitioning or sampling during training. A URL to a prime numbers search tool is provided to help find this prime number.

### 5. Training Configuration and Execution

- The script configures and initiates the training process using the `train.py` script with a wide array of parameters, including:
  - `--wandb "":` Disables Weights & Biases integration.
  - `--proj_dir`: Specifies the project directory for saving models and logs.
  - `--data_file`, `--data_type`, `--vocab_size`: Define the dataset path, its type (`binidx` for binary indexed dataset), and the vocabulary size.
  - `--ctx_len`: Sets the context length for the model.
  - `--my_pile_stage`: A custom parameter potentially used for controlling stages of training or dataset preprocessing.
  - `--epoch_count`, `--epoch_begin`, `--epoch_save`: Configures the number of epochs to train, the starting epoch, and saving frequency.
  - `--weight_decay`, `--head_size_a`, `--num_nodes`, `--micro_bsz`, `--n_layer`, `--n_embd`, `--pre_ffn`, `--head_qk`, `--my_exit_tokens`, `--magic_prime`: Further model and training parameters.
  - `--lr_init`, `--lr_final`, `--warmup_steps`: Learning rate configuration.
  - `--accelerator`, `--devices`, `--precision`, `--strategy`, `--grad_cp`, `--enable_progress_bar`, `--ds_bucket_mb`: Infrastructure and performance optimization settings, including the use of CPU, precision, distributed training strategy (DeepSpeed), and gradient checkpointing.

This script is a comprehensive example of how to configure and start training a complex neural network model like RWKV, providing a template for adjustments based on specific requirements or different datasets.

## Training Execution - `demo-training-run.sh' 

The `demo-training-run.sh` script is designed to configure and execute the training of the RWKV Language Model with specific parameters. It builds on the setup provided by `demo-training-prepare.sh`, adjusting for an actual training run with different settings. 

```bash
#!/bin/bash

BASE_NAME="model/0.1-1"
N_LAYER="12"
N_EMBD="768"
M_BSZ="16" # takes 16G VRAM (reduce this to save VRAM)
LR_INIT="6e-4"
LR_FINAL="6e-5"
GRAD_CP=0 # set to 1 to save VRAM (will be slower)
EPOCH_SAVE=10

# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case)
# use https://www.dcode.fr/prime-numbers-search

python train.py --load_model "0" --wandb "RWKV-5-Test" --proj_dir $BASE_NAME \
 --ctx_len 512 --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --num_nodes 1 --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb 200
```

### Model Configuration

- `BASE_NAME`, `N_LAYER`, `N_EMBD`: These variables specify the base directory for the model, the number of layers, and the embedding size, respectively. The model configuration remains the same as in the preparation script with 12 layers and an embedding size of 768.
- `M_BSZ`: Defines the micro batch size (number of samples processed at a time). The comment suggests that a batch size of 16 requires around 16GB of VRAM. Adjusting this value can help manage VRAM usage.
- `LR_INIT`, `LR_FINAL`: These variables set the initial and final learning rates for the training process.
- `GRAD_CP`: A flag for gradient checkpointing, which can be enabled (set to 1) to save VRAM at the cost of slower training speed.
- `EPOCH_SAVE`: Determines how frequently (in epochs) the model checkpoints are saved.

### Training Execution

- The script calls the `train.py` script with a set of parameters to start the training process. Notable parameters include:
  - `--load_model "0"`: Indicates the starting point for training. A value of "0" might refer to starting from scratch or a specific initial model setup.
  - `--wandb "RWKV-5-Test"`: Specifies the Weights & Biases project name for logging and tracking the training process.
  - `--data_file "data/minipile"`: Specifies the dataset to be used for training.
  - `--my_exit_tokens 1498226207`, `--magic_prime 2926181`: Custom parameters, with `magic_prime` as calculated for dataset partitioning or sampling.
  - `--num_nodes`, `--micro_bsz`, `--n_layer`, `--n_embd`, `--pre_ffn`, `--head_qk`, `--lr_init`, `--lr_final`, `--warmup_steps`, `--beta1`, `--beta2`, `--adam_eps`, `--my_pile_edecay`, `--data_type`, `--vocab_size`, `--weight_decay`, `--epoch_save`, `--head_size_a`: These parameters further define the model's architecture, training strategy, learning rate schedule, optimizer settings, dataset details, and regularization.
  - `--accelerator gpu`, `--devices 1`, `--precision bf16`, `--strategy deepspeed_stage_2`, `--enable_progress_bar True`, `--ds_bucket_mb 200`: Infrastructure and performance optimization settings, including the use of GPU acceleration, precision format, distributed training strategy (DeepSpeed), and enabling the progress bar for training monitoring.

This script provides a comprehensive setup for running the training of the RWKV Language Model, highlighting the flexibility and customization options available for model training, including adjustments for computational resources, learning rate scheduling, and integration with experiment tracking tools like Weights & Biases.

### Demo Data -- `demo.jsonl`

The `demo.jsonl` file contains JSON Lines format data, where each line is a separate JSON object. This format is particularly useful for handling large datasets of structured data and is commonly used in machine learning and natural language processing tasks for training models on conversational AI, text summarization, question-answering systems, and more.

```json lines
{"text": "System: You are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n\nUser: Summarize this article in one sentence.\nYou'll find this in your Start menu. You can also press ⊞ Win and type \"xbox.\" You can use the Xbox app to take a screenshot of any game running in Windows 10. You'll find this in the menu on the left side of the screen. This will display the DVR and screenshot settings. The Game DVR will need to be enabled in order to take screenshots with the app. Click the slider to toggle it on. When the Game DVR is enabled, the built-in shortcut is ⊞ Win+Alt+PrtScn. You can click the empty field next to this and set your own custom shortcut if you want. This will open the folder that your screenshots will be saved in. If you want to change where your screenshots are saved, you'll need to move the Captures folder to that location. Screenshots will automatically save to the Captures folder, wherever it is. For example, to move it to your Pictures folder, move up one directory and then drag the Captures folder to the Pictures folder in your sidebar. The Xbox app does not need to be open to use the screenshot feature. It will be saved to the Captures folder that you may have moved earlier.\n\nAssistant: The article explains how to use the Xbox app in Windows 10 to take screenshots of games by enabling the Game DVR and using the built-in shortcut or a custom shortcut, with the screenshots saved in the Captures folder which can be moved to a different location if desired."}
{"text": "User: Q:I read this background article the other day: Water stratification is when water masses with different properties - salinity (halocline), oxygenation (chemocline), density (pycnocline), temperature (thermocline) - form layers that act as barriers to water mixing which could lead to anoxia or euxinia.[1] These layers are normally arranged according to density, with the least dense water masses sitting above the more dense layers.  Water stratification also creates barriers to nutrient mixing between layers. This can affect the primary production in an area by limiting photosynthetic processes. When nutrients from the benthos cannot travel up into the photic zone, phytoplankton may be limited by nutrient availability. Lower primary production also leads to lower net productivity in waters.[2]  I am facing a new situation today: High lake had a lot of fresh and salt water mixing, therefore presented layers of stratified water.The water was poor in nutrients.Low lake had no such problems and it was a very abundant and prolific fishing spot.  Using the knowledge I acquired from the background article, how should I answer correctly the following question regarding my new situation: Which lake had fewer dissolved nutrients?\nA:\n\nAssistant: The lake with fewer dissolved nutrients would be High lake. This is because it had layers of stratified water, which acts as a barrier to nutrient mixing. As a result, it was poor in nutrients."}
...
```

Each JSON object in this file represents a dialog between a "User" and a "System" (the AI assistant), with the system's responses generated based on the user's queries. The excerpts provided illustrate two different scenarios:

### 1. Summarizing an Article on Taking Screenshots with the Xbox App in Windows 10

- **User Prompt**: The user asks to summarize an article in one sentence.
- **Article Content**: Describes the process of using the Xbox app on Windows 10 to take screenshots of games, including enabling the Game DVR, using shortcuts for screenshots, and managing the Captures folder where screenshots are saved.
- **AI Response**: The AI summarizes the article by stating that it explains how to use the Xbox app in Windows 10 to take game screenshots, highlighting the enabling of Game DVR, usage of shortcuts, and management of the Captures folder.

### 2. Answering a Question Based on Background Knowledge on Water Stratification

- **User Prompt**: After providing background information on water stratification, the user presents a scenario involving two lakes, High lake and Low lake, with different conditions of water mixing and nutrient availability. The user then asks which lake had fewer dissolved nutrients.
- **Background Information**: Details about water stratification, including definitions of terms like halocline, chemocline, pycnocline, and thermocline, and the implications of stratification on nutrient mixing and primary production.
- **New Situation**: High lake has mixed fresh and saltwater leading to stratification and nutrient-poor conditions, whereas Low lake does not have such problems and is abundant in fish.
- **AI Response**: Based on the background information and the described situation, the AI concludes that High lake would have fewer dissolved nutrients due to its stratified water layers acting as barriers to nutrient mixing.

These examples demonstrate how the dataset can be used to train AI models on tasks such as text summarization and applying learned knowledge to answer questions. The structured format of the data, with clear separation of prompts and responses, facilitates the training of models to understand and generate contextually appropriate answers.

### Creating Training Data - `make_data.py`

The `make_data.py` script is a Python utility designed to process and prepare datasets for training neural network models, specifically for tasks involving natural language processing (NLP). 

```python
import json, math, random, sys, time, shutil, os, string, re, fileinput
import numpy as np

"""
How to use:

python make_data.py demo.jsonl 3 4096

This will:
==> shuffle & duplicate demo.jsonl (for 3 epochs, good for finetuning) note: this will be very slow for large jsonl and we need more efficient code.
==> load jsonl and tokenize
==> save as demo.bin & demo.idx
==> compute "magic_prime" for ctxlen 4096

Example:

Assume your source jsonl is:
{"text":"aa"}
{"text":"bb"}
{"text":"cc"}
{"text":"dd"}

The final binidx will be like (here "/" means end_of_doc, which is actually token [0]):
bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/

where the data is repeated 3 times (each time with different shuffle)
"""

########################################################################################################
# MMapIndexedDatasetBuilder
########################################################################################################

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
from src.binidx import MMapIndexedDataset
def index_file_path(prefix_path):
    return prefix_path + ".idx"
def data_file_path(prefix_path):
    return prefix_path + ".bin"
class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
    def end_document(self):
        self._doc_idx.append(len(self._sizes))
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
cnt = 0
def add_raw(raw):
    global builder, cnt
    out = tokenizer.encode(raw)
    if tokenizer.decode(out) != raw:
        print("ERROR" * 100)
        exit(0)
    out.append(0)  # [0] = end_of_doc for rwkv tokenizer
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if cnt % 500 == 0:
        print(cnt, end=" ", flush=True)
    cnt += 1
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

########################################################################################################

N_EPOCH = int(sys.argv[2].strip())
IN_FILE = sys.argv[1].strip()
OUT_NAME = os.path.splitext(os.path.basename(IN_FILE))[0]
CTX_LEN = int(sys.argv[3].strip())
TEMP_FILE = "make_data_temp.jsonl"

print(f"### Convert {IN_FILE} to {OUT_NAME}.bin/idx...")

with open(IN_FILE, "r", encoding="utf-8") as file:
    non_empty_lines = [line.strip() for line in file if line.strip()]

print(f"### Found {len(non_empty_lines)} non-empty lines in {IN_FILE}")

file = open(TEMP_FILE, "w", encoding="utf-8")
for i in range(N_EPOCH):
    print(f"Shuffle: {i+1} out of {N_EPOCH}")
    random.shuffle(non_empty_lines)
    for entry in non_empty_lines:
        file.write(entry + "\n")
file.close()

########################################################################################################

print("### Building binidx...")

builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")
with fileinput.input(TEMP_FILE, encoding="utf-8") as ffff:
    for line in ffff:
        x = json.loads(line)["text"]
        add_raw(x)
builder.finalize((f"{OUT_NAME}.idx"))
print("done")

print("### Verifying result...")
data = MMapIndexedDataset(OUT_NAME)
data_len = len(data)
data_size = len(data._bin_buffer) // data._index._dtype_size

TODO = [0, data_len - 1]
PREVIEW_LIMIT = 100
for idx in TODO:
    ptr, size = data._index[idx]
    dix = data.get(idx=idx, offset=0, length=size).astype(int)
    print("-" * 70 + f"[{OUT_NAME} idx {idx} sz {size}]")
    assert dix[-1] == 0
    dix = dix[:-1]
    if len(dix) > PREVIEW_LIMIT:
        try:
            print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
        except:
            try:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 1]))
            except:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 2]))
        print("· " * 30)
        try:  # avoid utf-8 bug
            print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
        except:
            try:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1 :]))
            except:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 2 :]))
    else:
        print(tokenizer.decode(dix))

print(f"{'-'*80}\n### Final {OUT_NAME}.bin/idx has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")

if data_size >= CTX_LEN * 3:
    n_chunk = int(data_size // CTX_LEN) - 1
    for i in range(n_chunk, 0, -1):
        if i % 3 == 2:
            if is_prime(i):
                print(f"\n### magic_prime = {i} (for ctxlen {CTX_LEN})\n")
                exit(0)
```

1. **Shuffle and Duplicate Data**: For each epoch specified, the script shuffles the lines in the input `.jsonl` file and duplicates them. This process is useful for fine-tuning models on relatively small datasets by artificially increasing the dataset size and introducing variability.

2. **Tokenize Data**: It reads the shuffled and duplicated `.jsonl` data, tokenizes the text using a specified tokenizer, and then converts the tokenized data into a binary format suitable for efficient storage and access during model training.

3. **Save Binary and Index Files**: The tokenized data is saved as `.bin` (binary data file) and `.idx` (index file) files. These files are used by models to quickly load and process data in batches during training.

4. **Compute `magic_prime`**: The script calculates a prime number, referred to as `magic_prime`, based on the context length (`ctxlen`) and the size of the processed dataset. This prime number might be used for sampling or partitioning the dataset in a specific manner during training.

### Detailed Steps:

- **Preparing Temporary Files**: Creates a temporary `.jsonl` file to hold the shuffled data across epochs.
- **Tokenization and Data Writing**: Utilizes a custom tokenizer (e.g., `TRIE_TOKENIZER`) to encode the raw text data into token IDs, appending a special end-of-document token to each document. The token IDs are written to the binary data file, and document boundaries are marked in the index file.
- **Finalization**: After processing all data, the script closes the binary file and writes the index file, finalizing the dataset preparation.
- **Verification**: Loads the created dataset to verify its integrity and prints a preview of the data from the beginning and end of the dataset for inspection.
- **Magic Prime Calculation**: Calculates the `magic_prime` value based on the total number of tokens in the dataset and the specified context length, ensuring the prime number satisfies certain conditions (e.g., `3n+2` form and being smaller than a certain threshold).

### Usage Example:

```
python make_data.py demo.jsonl 3 4096
```

- This command processes `demo.jsonl`, shuffles and duplicates it for 3 epochs, tokenizes the data, saves it as `demo.bin` and `demo.idx`, and computes `magic_prime` for a context length of 4096.

This script is an essential tool for data preprocessing in machine learning workflows, enabling efficient data manipulation, tokenization, and storage for large-scale NLP tasks.

## Train Model - `train.py`

The `train.py` script is a comprehensive training utility for the RWKV (Recurrent Weighted Key Value) Language Model, leveraging the PyTorch Lightning framework to streamline the training process of deep learning models, particularly in natural language processing tasks. This script is structured to handle various aspects of model training, including model architecture specification, data loading, training configuration, and integration with the Weights & Biases (wandb) tool for experiment tracking.

```python
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='', type=str)
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    if pl.__version__[0]=='2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    else:
        parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    if args.data_type == "wds_img":
        args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
        args.proj_dir = f"{args.proj_dir}-{args.run_name}"
    else:
        args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime

        if args.my_pile_shift < 0:
            args.my_pile_shift = 0

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak
        if args.my_qa_mask == 2:
            args.epoch_count = 2 * args.magic_prime // 40320
        else:
            args.epoch_count = args.magic_prime // 40320

        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        # if args.my_pile_stage == 2:
        #     assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p != "final":
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
            list_p.sort()
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.warmup_steps < 0:
                    if args.my_pile_stage == 2:
                        args.warmup_steps = 10
                    else:
                        args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    from src.model import RWKV
    model = RWKV(args)

    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.my_pile_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
    model.load_state_dict(load_dict)

    if pl.__version__[0]=='2':
        trainer = Trainer(accelerator=args.accelerator,strategy=args.strategy,devices=args.devices,num_nodes=args.num_nodes,precision=args.precision,
        logger=args.logger,callbacks=[train_callback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
        )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            else:
                print(f"{str(shape[0]).ljust(5)}       {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    trainer.fit(model, data_loader)
```

### Key Features and Configuration Options:

1. **Model and Training Configuration**: Through the use of command-line arguments, it allows for detailed configuration of the model (e.g., number of layers, embedding dimension, attention dimension), training parameters (e.g., learning rate, batch size, epoch count), and data handling (e.g., data file location, data type, vocabulary size).

2. **Support for Different Data Types**: It supports various data types for training, including text (with different encodings), numpy arrays, and specifically formatted binary index data (`binidx`), allowing flexibility in handling diverse datasets.

3. **Gradient Accumulation and Precision Control**: Offers options for gradient checkpointing to save VRAM, dropout settings for regularization, and precision control (e.g., FP32, FP16, BF16) to balance between computational speed and memory usage.

4. **Dynamic Learning Rate and Weight Decay**: Implements a customizable learning rate schedule, including warmup steps and options for setting initial and final learning rates, as well as weight decay for regularization.

5. **PyTorch Lightning Integration**: Utilizes PyTorch Lightning for managing the training loop, device allocation (GPU/CPU), distributed training, and precision settings, making the training process more efficient and scalable.

6. **Custom Callbacks and Logging**: Includes a custom callback (`train_callback`) for additional training logic such as learning rate adjustments, model saving, and logging via Weights & Biases for tracking the training process and analyzing experiments.

7. **Load Pre-trained Models**: Supports loading pre-trained models for fine-tuning or continuing training, with options for partial loading if the model architecture changes.

8. **Deepspeed Integration**: Offers integration with Deepspeed for efficient distributed training, optimizing training speed and resource usage on multiple GPUs or nodes.

### Execution Flow:

- Parses command-line arguments to configure the model, data, and training parameters.
- Initializes logging and seed for reproducibility.
- Prepares the dataset by instantiating a `MyDataset` object based on the specified data file and type.
- Constructs the RWKV model with the given configuration and optionally loads a pre-trained model.
- Sets up a PyTorch Lightning `Trainer` with the specified training strategy, devices, and precision.
- Prepares a `DataLoader` for feeding data to the model during training.
- Starts the training process using the trainer, model, and data loader.

### Usage:

The script is executed from the command line with various options to specify model dimensions, data paths, training hyperparameters, and more. For example:

```bash
python train.py --data_file "path/to/data_file" --n_layer 12 --n_embd 768 --epoch_count 100 --lr_init 1e-4
```

This command starts the training of the RWKV model with specified layers, embedding dimensions, and learning rate on the given dataset.

## Running the Eagle 7B Language Model

For various methods to run the model, consult the official blog for guidance.

https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers

For our purposes, we'll employ a direct approach to executing the model. Initially, you must download the model weights from Hugging Face:

https://huggingface.co/RWKV/v5-Eagle-7B/blob/main/RWKV-v5-Eagle-World-7B-v2-20240128-ctx4096.pth

You don't need a separate tokenizer as we'll be using an inference package from the RWKV team.

```bash
pip install rwkv
```

Then you can run the model using the following code:

```python
# set these before import RWKV
import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################

MODEL_PATH = "E:/models/Eagle-7B/RWKV-v5-Eagle-World-7B-v2-20240128-ctx4096.pth"
TOKENIZER_PATH = "E:/models/Eagle-7B/20B_tokenizer.json"

model = RWKV(model=MODEL_PATH, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # rwkv "world" models - included in the rwkv package

ctx = "\nAttention Free Transformer is"
print(ctx, end='')


def my_print(s):
    print(s, end='', flush=True)


args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
print('\n')

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')
```

For convenience, two script versions are provided, tailored to different hardware configurations:

- **`run-inference-cuda.py`**: This script was utilized on a Windows system equipped with an RTX 4090 24GB GPU, applying the `cuda fp16` strategy. This strategy optimizes the GPU's capabilities for enhanced computational speed.
  
- **`run-inference-mac.py`**: This script was executed on an Apple M2 Ultra, which boasts 192GB of RAM, using the `cpu fp32` strategy. This approach leverages the CPU for model execution.

The primary differences between these scripts lie in the computational strategy (GPU vs. CPU) and the specified paths for accessing the model and tokenizer files.

Please note that this model is designed for text generation and not specifically tailored for chat applications or completing sentences as is. For guidance on running chat models or using UIs, please refer to the following repos:

👉 https://github.com/BlinkDL/ChatRWKV

👉 https://github.com/BlinkDL/RWKV-LM

### Environment Configuration

The first part of the code sets environment variables to control the behavior of the RWKV library:

- `RWKV_JIT_ON`: Enables JIT (Just-In-Time) compilation for the RWKV model, potentially improving performance by compiling Python functions into machine code at runtime.
- `RWKV_CUDA_ON`: Controls whether to compile and use CUDA kernels for GPU acceleration. Setting this to '1' would significantly speed up computations on compatible NVIDIA GPUs, assuming all necessary dependencies are installed.

### Model and Tokenizer Initialization

- The model is loaded from a specified path using the `RWKV` class. The `strategy` parameter indicates that CUDA with FP16 (16-bit floating-point) precision should be used, which is suitable for GPUs and helps to reduce memory consumption and potentially increase computation speed.
- The tokenizer, which is essential for converting text into tokens (integers) that the model can understand, is specified but not directly used in the shown code snippet.

When I ran the model on my Apple M2 Ultra equipped with 192GB RAM using `cpu 32` settings, I observed slow text generation times. However, on a Windows machine equipped with an RTX 4090 24GB GPU, the performance significantly improved, leading to much faster text generation. According to the RWKV team, compiling the custom CUDA kernel could further enhance performance, suggesting an even faster processing capability.

Ensure that the `MODEL_PATH` and `TOKENIZER_PATH` variables are correctly set to the paths where your model and tokenizer files are located on your system. We're utilizing the "world" model identified by `rwkv_vocab_v20230424`, which is conveniently included within the RWKV package. Using the following defaults without proper adjustment might lead to unexpected and incorrect results:

```python
pipeline = PIPELINE(model, "20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
```

With the world model:

```python
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # rwkv "world" models - included in the rwkv package
```

### Text Generation Pipeline:

- A pipeline for generating text is created using the `PIPELINE` utility, which wraps around the model to facilitate text generation tasks.
- `PIPELINE_ARGS` is used to configure various parameters for the generation process, such as temperature, top_p, and top_k for controlling the randomness and diversity of generated text, as well as other parameters for penalizing or favoring certain tokens during generation.

### Text Generation:

- The text generation is initiated with a given context (`ctx`) and a specified number of tokens to generate. The generation process is customized with the provided arguments (`args`). The `callback` function `my_print` is used to print generated tokens continuously without introducing new lines, providing a seamless output.

### Model Forward Pass:

- Demonstrates how to perform a forward pass through the model using both single-shot and stateful approaches. This part of the code illustrates how the model can be used in an RNN-like manner, maintaining state across calls to generate coherent sequences of logits (unnormalized probabilities) based on input tokens.
- The `state` parameter is used to carry over hidden states between calls to `model.forward()`, allowing the model to maintain context and produce consistent output across sequential inputs.

## Final Thoughts

Through our detailed exploration of the RWKV Language Model and its suite of tools, we've developed a thorough understanding of its architecture, training methodologies, and application in generating text. The RWKV model stands out as a fascinating alternative to conventional transformer models, offering a unique blend of flexibility, efficiency, and the possibility for further enhancements.

However, as emphasized at the outset of our exploration, the ultimate evaluation of the RWKV model's potential and impact is left to users and the broader open-source community. The evolutionary principles of survival of the fittest and natural selection apply as much in the realm of technology as they do in biology. Should the RWKV models prove to be the optimal solution, they will undoubtedly prosper and expand. On the other hand, should they not meet expectations, they will be surpassed by more advanced innovations.

One of the primary challenges that the RWKV model faces is its relative obscurity compared to the well-established Transformer models, which have become the de facto standard in the field of natural language processing. Additionally, the RWKV model's roots in RNN architectures—a technology many consider to be from a bygone era in AI—could pose another obstacle in its path to widespread acceptance. However, if the RWKV model truly holds the potential that some proponents claim, its ascent to prominence may just be a matter of time.

The journey of RWKV from obscurity to recognition will hinge on its ability to demonstrate clear advantages over existing models. These could include improved efficiency, adaptability, or performance in specific tasks. As the AI community continues to explore and push the boundaries of what's possible, innovative models like RWKV have the opportunity to reshape our understanding and application of machine learning algorithms.

Ultimately, the future of the RWKV model will be determined by its adoption and success in practical applications. If it can navigate the initial hurdles of unfamiliarity and skepticism by proving its value, it may well establish itself as a new norm in the ever-evolving landscape of artificial intelligence technologies. 

For us, this journey into the depths of AI and machine learning has been yet another enriching learning experience. And in the end, that's what truly counts.



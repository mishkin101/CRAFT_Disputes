import unicodedata
import json
import os
import regex
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import emoji

import model.config as cfg
from model.data import *

VOCAB_SIZE = 50000
NUM_CHUNKS = 200

def detect_emojis(tok):
    return tok in emoji.EMOJI_DATA and not tok.isdigit()

def count_tokens_in_chunk(idx, chunk):
    print("Processing chunk", idx)
    counts = Counter()

    for dialog in chunk:
        for utterance in dialog:
            tokens = tokenize(utterance["text"])
            for tok in tokens:
                # Remove emojis
                if detect_emojis(tok):
                    tok = "UNK"  # Replace with UNK token
            counts += Counter(tokens)
    return counts

def build_Vocab():
    print("Loading training dataset...")
    with open(os.path.join(cfg.data_dir, "nn_input_data", cfg.corpus_name, "train_processed_dialogs.txt")) as fp:
        dialogs = [json.loads(line) for line in fp]
    
    chunk_size = len(dialogs) // NUM_CHUNKS
    dialog_chunks = [dialogs[i:i+chunk_size] for i in range(0, len(dialogs), chunk_size)]

    global_counts = Counter()

    with Pool(40) as p:
        counts_per_dialog = p.starmap(count_tokens_in_chunk, list(enumerate(dialog_chunks)))
    print("Merging chunks...")
    for dialog_counts in counts_per_dialog:
        global_counts += dialog_counts

    print("Truncating to vocabulary size", VOCAB_SIZE)
    kept_counts = global_counts.most_common(VOCAB_SIZE)

    print("Converting to dicts")
    word2index = {"UNK": UNK_token}
    index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
    num_words = 4  # Count SOS, EOS, PAD, UNK
    for token, _ in kept_counts:
        word2index[token] = num_words
        index2word[num_words] = token
        num_words += 1
    
    print("Dumping")
    with open(cfg.word2index_path, "w") as fp:
        json.dump(word2index, fp)
    with open(cfg.index2word_path, "w") as fp:
        json.dump(index2word, fp)
    print(f"Vocab file written to {word2index_path}")

if __name__ == "__main__":
    build_Vocab()
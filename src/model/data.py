import torch
import nltk
import itertools
import random
import json
import unicodedata
from convokit import Corpus
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.utils import resample
import numpy as np 
import pandas as pd

from .config import *

class Voc:
    def __init__(self, name, word2index=None, index2word=None):
        self.name = name
        self.trimmed = False if not word2index else True # if a precomputed vocab is specified assume the user wants to use it as-is
        self.word2index = word2index if word2index else {"UNK": UNK_token}
        self.word2count = {}
        self.index2word = index2word if index2word else {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4 if not index2word else len(index2word)  # Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"UNK": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Tokenize the string using NLTK
def tokenize(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'\w+|[^\w\s]')
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text.lower())

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []
    
    return tokenizer.tokenize(cleaned_text)

# Create a Voc object from precomputed data structures
def loadPrecomputedVoc(corpus_name, word2index_path, index2word_path):
    with open(word2index_path) as fp:
        word2index = json.load(fp)
    with open(index2word_path) as fp:
        index2word = json.load(fp)
    return Voc(corpus_name, word2index, index2word)

# Given a dialog entry, consisting of a list of {text, label} objects, preprocess
# each utterance's text by tokenizing and truncating.
# Returns the processed dialog entry where text has been replaced with a list of
# tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
def processDialog(voc, dialog):
    processed = []
    for utterance in dialog:
        tokens = tokenize(utterance["text"])
        if len(tokens) >= MAX_LENGTH:
            tokens = tokens[:(MAX_LENGTH-1)]
        # replace out-of-vocabulary tokens
        for i in range(len(tokens)):
            if tokens[i] not in voc.word2index:
                tokens[i] = "UNK"
        processed.append({"tokens": tokens, "is_attack": utterance.get("labels", {}).get("is_attack", None), "convo_id": utterance.get("labels", {}).get("id", None)})
    return processed

# Load context-reply pairs from the given dataset.
# Since the dataset may be large, we avoid keeping more data in memory than
# absolutely necessary by cleaning each utterance (tokenize, truncate, replace OOV tokens)
# line by line in this function.
# Returns a list of pairs in the format (context, reply, label)
def loadPairs(voc, path, last_only=False):
    pairs = []
    with open(path) as datafile:
        for i, line in enumerate(datafile):
            print("\rLine {}".format(i+1), end='')
            raw_convo_data = json.loads(line)
            dialog = processDialog(voc, raw_convo_data)
            iter_range = range(1, len(dialog)) if not last_only else [len(dialog)-1]
            for idx in iter_range:
                reply = dialog[idx]["tokens"][:(MAX_LENGTH-1)]
                label = dialog[idx]["is_attack"]
                convo_id = dialog[idx]["convo_id"]
                # gather as context up to CONTEXT_SIZE utterances preceding the reply
                start = max(idx - CONTEXT_SIZE, 0)
                context = [u["tokens"] for u in dialog[start:idx]]
                pairs.append((context, reply, label, convo_id))
        print()
    return pairs

# Using the functions defined above, return a list of pairs for unlabeled training
def loadUnlabeledData(voc, train_path):
    print("Start preparing training data ...")
    print("Preprocessing training corpus...")
    train_pairs = loadPairs(voc, train_path)
    print("Loaded {} pairs".format(len(train_pairs)))
    return train_pairs

# Using the functions defined above, return a list of pairs for labeled training
def loadLabeledData(voc, attack_train_path, attack_val_path, analysis_path):
    print("Start preparing training data ...")
    print("Preprocessing labeled training corpus...")
    attack_train_pairs = loadPairs(voc, attack_train_path, last_only=True)
    print("Loaded {} pairs".format(len(attack_train_pairs)))
    print("Preprocessing labeled validation corpus...")
    attack_val_pairs = loadPairs(voc, attack_val_path, last_only=True)
    print("Loaded {} pairs".format(len(attack_val_pairs)))
    print("Preprocessing labeled analysis corpus...")
    analysis_pairs = loadPairs(voc, analysis_path, last_only=True)
    print("Loaded {} pairs".format(len(analysis_pairs)))
    return attack_train_pairs, attack_val_pairs, analysis_pairs

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(False)
            else:
                m[i].append(True)
    return m

# Takes a batch of dialogs (lists of lists of tokens) and converts it into a
# batch of utterances (lists of tokens) sorted by length, while keeping track of
# the information needed to reconstruct the original batch of dialogs
def dialogBatch2UtteranceBatch(dialog_batch):
    utt_tuples = [] # will store tuples of (utterance, original position in batch, original position in dialog)
    for batch_idx in range(len(dialog_batch)):
        dialog = dialog_batch[batch_idx]
        for dialog_idx in range(len(dialog)):
            utterance = dialog[dialog_idx]
            utt_tuples.append((utterance, batch_idx, dialog_idx))
    # sort the utterances in descending order of length, to remain consistent with pytorch padding requirements
    utt_tuples.sort(key=lambda x: len(x[0]), reverse=True)
    # return the utterances, original batch indices, and original dialog indices as separate lists
    utt_batch = [u[0] for u in utt_tuples]
    batch_indices = [u[1] for u in utt_tuples]
    dialog_indices = [u[2] for u in utt_tuples]
    return utt_batch, batch_indices, dialog_indices

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, already_sorted=False):
    if not already_sorted:
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch, label_batch, id_batch = [], [], [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        label_batch.append(pair[2])
        id_batch.append(pair[3])
    dialog_lengths = torch.tensor([len(x) for x in input_batch])
    input_utterances, batch_indices, dialog_indices = dialogBatch2UtteranceBatch(input_batch)
    inp, utt_lengths = inputVar(input_utterances, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    label_batch = torch.FloatTensor(label_batch) if label_batch[0] is not None else None
    return inp, dialog_lengths, utt_lengths, batch_indices, dialog_indices, label_batch, id_batch, output, mask, max_target_len

def batchIterator(voc, source_data, batch_size, shuffle=True):
    cur_idx = 0
    if shuffle:
        random.shuffle(source_data)
    while True:
        if cur_idx >= len(source_data):
            cur_idx = 0
            if shuffle:
                random.shuffle(source_data)
        batch = source_data[cur_idx:(cur_idx+batch_size)]
        # the true batch size may be smaller than the given batch size if there is not enough data left
        true_batch_size = len(batch)
        # ensure that the dialogs in this batch are sorted by length, as expected by the padding module
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # for analysis purposes, get the source dialogs and labels associated with this batch
        batch_dialogs = [x[0] for x in batch]
        batch_labels = [x[2] for x in batch]
        # convert batch to tensors
        batch_tensors = batch2TrainData(voc, batch, already_sorted=True)
        yield (batch_tensors, batch_dialogs, batch_labels, true_batch_size) 
        cur_idx += batch_size


"=========================CUSTOM FUNCTIONS===================================="
def createTrainFile():
    out_dir = os.path.join(data_dir, "nn_input_data", corpus_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train_processed_dialogs.txt")
    
    with open(out_path, "w") as fp:
        for entry in os.listdir(corpus_dir):
            corpus_path = os.path.join(corpus_dir, entry)
            if not os.path.isdir(corpus_path):
                continue

            print(f"Pre‐training: loading {entry} …")
            ck_corpus = Corpus(filename=corpus_path)

            # for each conversation, pull every root→leaf reply‐chain
            for convo in ck_corpus.iter_conversations():
                for dialog in convo.get_root_to_leaf_paths():
                    dialog_json = []
                    for utt in dialog:
                        if utt.text not in pretrain_exclude_phrases:
                            dialog_json.append({"text": utt.text})
                    fp.write(json.dumps(dialog_json) + "\n")

    print(f"Pre‐training file written to {out_path}")


"custom Pre-training/fine-tuning split for Kodis."
"Add meta.pre-train tag to the conversations that are used for pre-training."
"need to determine size of test-set"
def createTrainTestSplit(convo_df):
    convo_ids = np.array(convo_df.index.tolist())
    convo_labels = convo_df[f"meta.{label_metadata}"].astype(int).values 
    if Imbalance_handling == 'default' | "downsampling":
        X_train, X_test, y_train, y_test = train_test_split(
            convo_ids, convo_labels, test_size= 1-val_size-train_size, random_state=random_seed)
    if Imbalance_handling == 'stratified':
        X_train, X_test, y_train, y_test = train_test_split(
            convo_ids, convo_labels, test_size= 1-val_size-train_size, stratify=convo_labels, random_state=random_seed)
    return X_train, X_test, y_train, y_test

"""Only do train/val splits on training set"""
def createTrainValSplit(convo_df_train):
    convo_ids    = np.array(convo_df_train.index.tolist())
    convo_labels = convo_df_train[f"meta.{label_metadata}"].astype(int).values
    if k_folds > 1:
        if Imbalance_handling == "stratified":
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            folds = []
            for train_idx, val_idx in skf.split(convo_ids, convo_labels):
                train_ids = convo_ids[train_idx]
                val_ids   = convo_ids[val_idx]
                folds.append((train_ids, val_ids))
            return folds
        if Imbalance_handling == "downsample":
            kf = KFold(n_splits=k_folds, shuffle = True, random_state = random_seed)
            for train_idx, val_idx in skf.split(convo_ids, convo_labels):
                train_ids = convo_ids[train_idx]
                train_ids_down = downsample(convo_df_train, train_ids)
                val_ids   = convo_ids[val_idx]
                folds.append((train_ids_down, val_ids))
            return folds
    else:
        test_size = val_size  
        train_ids, val_ids = train_test_split(convo_ids, test_size=test_size, random_state=random_seed)
        return [(train_ids, val_ids)]

def assignSplit(convo_df, train_ids, test_ids= None, val_ids = None):
    df = convo_df.copy()
    df["meta.split"] = None
    df.loc[train_ids, "meta.split"] = "train"
    if val_ids is not None:
        df.loc[val_ids,   "meta.split"] = "val"
    if test_ids is not None:
        df.loc[test_ids,  "meta.split"] = "test"
    return df

def downsample(convo_df, train_ids):
    train_ids = np.array(train_ids)
    train_labels = (convo_df.loc[train_ids, f"meta.{label_metadata}"].astype(int).to_numpy())
    ids_class0 = train_ids[train_labels == 0]
    ids_class1 = train_ids[train_labels == 1]
    if len(ids_class0) > len(ids_class1):
        majority_ids = ids_class0
        minority_ids = ids_class1
    else:
        majority_ids = ids_class1
        minority_ids = ids_class0

    n_minority = len(minority_ids)
    majority_down = resample(majority_ids ,replace=False, n_samples=n_minority,random_state=random_seed)
    downsampled_ids = np.concatenate([minority_ids, majority_down])
    convo_down = convo_df.loc[downsampled_ids]
    convo_down_ids = convo_down.index
    return convo_down_ids


def processLabeledDialogs(utt_df, voc):
    utt_df = utt_df.copy()
    utt_df["tokens"] = (utt_df["text"].map(tokenize) #-> tokens for eack text
        .map(lambda toks: [t if t in voc.word2index else "UNK" for t in toks])
    )
    if utt_label_metadata is None:
        utt_df["is_attack"] = 0
    else:
        utt_df["is_attack"] = (utt_df[f"meta.{utt_label_metadata}"].fillna(0).astype(int))
    return utt_df

def loadLabeledPairs(voc, utt_df, conv_df, last_only, split_key):
    utts = processLabeledDialogs(utt_df, voc)
    # Prepare split IDs
    splits = {}
    ids = conv_df.index[conv_df["meta.split"] == split_key].unique()
    splits[split_key] = set(ids)

    """***//TODO: #2 make this more generic, so that it can be used for anytpe of label***"""
    def make_pairs_for_split(ids_set, last_only):
        pairs = []
        for convo_id, dialog_df in utts[utts["conversation_id"].isin(ids_set)].groupby("conversation_id", sort=False):
            dialog_df['label'] = dialog_df.index
            dialog = dialog_df[["tokens", "is_attack", 'label']].to_dict("records")
            #If we only have conversation-level labels, we assume the entire dialog 
            # is a context since it does not include a derailment event. therefore, last comment needs to be encoded
            if utt_label_metadata is None:
                if label_metadata is None:
                    raise ValueError("If utt_label_metadata is None, label_metadata must be provided to identify the conversation label.")
                    return
                last_id = dialog[-1]["label"]
                conv_label = conv_df.at[convo_id, f"meta.{label_metadata}"]
                dialog.append({"tokens": ["UNK"],"is_attack": conv_label, "label": f"{last_id}"})
            idxs = [len(dialog) - 1] if last_only else range(1, len(dialog))
            for idx in idxs:
                context = [u["tokens"][: MAX_LENGTH - 1] for u in dialog[max(0, idx - CONTEXT_SIZE) : idx]]
                reply     = dialog[idx]["tokens"][: MAX_LENGTH - 1]
                label     = conv_df.at[convo_id, f"meta.{label_metadata}"] if label_metadata else dialog[idx]["is_attack"]
                # We make a forecast on the comment prior to the reply, so we use the label of the previous comment.
                comment_id = dialog[idx - 1]["label"]
                pairs.append((context, reply, label, comment_id))
        return pairs
    pairs = make_pairs_for_split(splits["train"], last_only)
    return pairs


def updateUtterances(utt_df, results):
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index.name = "comment_id"
    results_df = results_df.rename(columns=lambda c: f"meta.{c}")
    return utt_df.merge(
        results_df,
        how="right",
        left_index=True,
        right_index=True
    )

def updateConvos(convo_df, results):
    conv_acc = {}
    for comment_id, res in results.items():
        if "_" not in comment_id:
            continue
        conv_id = comment_id.rsplit("_", 1)[1]
        if conv_id not in conv_acc:
            conv_acc[conv_id] = {"probs": [], "preds": []}
        conv_acc[conv_id]["probs"].append(res["probability"])
        conv_acc[conv_id]["preds"].append(res["prediction"])
    rows = []
    for conv_id, vals in conv_acc.items():
        max_prob = max(vals["probs"])
        any_pred = 1 if any(p == 1 for p in vals["preds"]) else 0
        rows.append({"conversation_id": f"utt0_{conv_id}",
                     "meta.forecast_score": max_prob,
                     "meta.forecast_prediction": any_pred})
    conv_results_df = pd.DataFrame(rows).set_index("conversation_id")

    return convo_df.merge(
        conv_results_df,
        how="right",
        left_index=True,
        right_index=True
    )
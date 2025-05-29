from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


# import all configuration variables
from model.config import *
# import data preprocessing functions
from model.data import *
# import our custom PyTorch modules
from model.model import *


DEFAULT_CONFIG = {
    "dropout": 0.1,
    "batch_size": 64,
    "clip": 50.0,
    "learning_rate": 1e-5,
    "print_every": 10,
    "finetune_epochs": 30,
    "validation_size": 0.2,
}
def adapative_learning(optimizers_list, patience=3, factor=0.5):
        return

def processDialog(voc, dialog):
    processed = []
    for utterance in dialog.iter_utterances():
        # skip the section header, which does not contain conversational content
        if corpus_name == 'wikiconv' and utterance.meta['is_section_header']:
            continue
        tokens = tokenize(utterance.text)
        # replace out-of-vocabulary tokens
        for i in range(len(tokens)):
            if tokens[i] not in voc.word2index:
                tokens[i] = "UNK"
        processed.append({"tokens": tokens, "is_attack": int(utterance.meta[utt_label_metadata]) if utt_label_metadata is not None else 0, "id": utterance.id})
    if utt_label_metadata is None:
        # if the dataset does not come with utterance-level labels, we assume that (as in the case of CMV)
        # the only labels are conversation-level and that the actual toxic comment was not included in the
        # data. In that case, we must add a dummy comment containing no actual text, to get CRAFT to run on 
        # the context preceding the dummy (that is, the full prefix before the removed comment)
        processed.append({"tokens": ["UNK"], "is_attack": int(dialog.meta[label_metadata]), "id": processed[-1]["id"] + "_dummyreply"})
    return processed

""" 
1.Data Pre-processing: 

    processDialog() *** NEED TO MODIFY ***
        -  tokenization of texts in eatch batch 

    loadPairs() *** NEED TO MODIFY ***
        -   loading pairs according to the split they are in as (context, reply, label, comment_id). 
        -   Called by processDialog()

    contextSelection():  *** NEED TO MAKE ***
        - which utterance to exlcude as context from meta.text

2.Loading Corpus Objects: 
    loadCorpusObjects() *** NEED TO MAKE ***
        -  load the corpus objects from corpus_dir
        -  load the utterances, speakers, and conversations dataframes
        -  this will be used to get the utterances and their metadata for training and evaluation
    load all corpora into memory

3.Load Vocabulary Object:
    loadPrecomputedVoc()

4.Build Classifiers:
    NEW Predictor(nn.Module): - Subjecitve outcomes *** NEED TO MAKE ***
    Predictor(nn.Module): - Objective outcomes (single target) Exists in model.py

5. Build Predictor Harness:(used in test/eval tests)
    - runs through utt encoder + contetext encoder to create hidden states
    - calls classifier to get predicitons from encoded context

6. Build Train Loop: (for fine-tuning training)
     src.train()
    - zero out gradients on every mini-batch
    - runs through utt encoder + contetext encoder to create hidden states
    - calculate loss
    - clip gradients to prevent exploding
    - adjust weights
    - return loss

7. Build evaluateBatch:
    src.evaluateBatch() *** NEED TO MODIFY ***
    - returns the (predicitons) and scores for test/val tests

8. Build validate
    src.validate() *** NEED TO MODIFY ***

9. Managing saving experiment results
    src.file_utils.save_experiment_results() *** NEED TO MAKE ***
    - save model state dict
    - save training history
    - save validation history
    - save test history
    - save hyper-parameters used in the experiment
    - save model architecture (if changed)
    - save final utterances, conversations dataframes to experiment in "experiments" directory

10. Plotting Utililties
 *** FIND GOOD PACKAGE TO MANAGE MONITORING TRAINING ***
    - plot training history
    - plot validation history
    - plot test history
    - plot hyper-parameters used in the experiment
    - save plots to "experiments" directory

12. build hyper-parameter tuning with parallelization
 *** FIND GOOD PACKAGE TO MANAGE MONITORING TRAINING ***
    - use Optuna to tune hyper-parameters and parallelize
    - save best hyper-parameters to "experiments" directory

13. us MLFlow to manage the training pipeline
    - use MLFow to track experiments, hyper-parameters, and results
    - save model artifacts to MLFow server
    - log training history, validation history, and test history to MLFow server
 *** FIND GOOD PACKAGE TO MANAGE MONITORING TRAINING ***
    - use Optuna to tune hyper-parameters and parallelize
    - save best hyper-parameters to "experiments" directory

11. Build and create Fine-tuning harness:
    - add command line flags to run an experiment confirguation with:
        - experiment name 
        - used for saving results in "experiments" directory
        - Torch model parameters to choose with Optuna:
            - fine_tuning epochs
            - batch size
            - learning rate
            - dropout rate
            - validation size
        - random seed that controls:
            - data in train/val/test splits
            - stratified sampling
            - undersampling
        - model dataset balance strategy
            - undersampling (downsampling majority)
            - stratified sampling
            - none
        - Utterance (context) selection strategy
            - submit agreement boxes
            - last utterance 
            - custom utterance selection function


"""

if __name__ == "__main__":
    


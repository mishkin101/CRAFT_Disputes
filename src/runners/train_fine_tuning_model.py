from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


# import all configuration variables
from model.config import *
# import data preprocessing functions
from model.data import *
# import CRAFT models
from model.model import *
# import custom classifier heads
from model.classifiers import *
#import optimizer
from model.optimizer import *


DEFAULT_CONFIG = {
    "dropout": 0.1,
    "batch_size": 64,
    "clip": 50.0,
    "learning_rate": 1e-5,
    "print_every": 10,
    "finetune_epochs": 30,
    "validation_size": 0.2,
}

"""Load Device"""
def loadDevice(type='cuda'):
    if type == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

"""Load Pretrained Model"""
def loadPretrainedModel(device):
    model_path = os.path.join(save_dir, "model.tar")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if device.type == 'cuda':
        print("Loading model on GPU")
        checkpoint = torch.load(model_path)
    else:
        print("Loading model on CPU")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    return checkpoint

"""Create Classifier Head"""
def createClassifierHead(class_type = 'single_target'):
    if class_type == 'single_target':
        return SingleTargetClf(hidden_size, dropout)
    return

"""Build Contect encoder, decoder, and classifier"""
def loadCheckpoint(checkpoint, device, classifier_type = 'single_target'):
    # Instantiate your modules
    voc             = loadPrecomputedVoc(corpus_name, word2index_path, index2word_path)
    embedding       = nn.Embedding(voc.num_words, hidden_size)
    encoder         = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
    attack_clf      = SingleTargetClf(hidden_size, dropout)
    voc.__dict__    = (checkpoint['voc_dict'])
    attack_clf      = createClassifierHead(classifier_type)
    # Load weights
    embedding.load_state_dict(checkpoint['embedding'])
    encoder.load_state_dict(checkpoint['en'])
    context_encoder.load_state_dict(checkpoint['ctx'])
    voc.__dict__ = checkpoint['voc_dict']
    # Move to device
    encoder         = encoder.to(device)
    context_encoder = context_encoder.to(device)
    attack_clf      = attack_clf.to(device)

    #set to train mode
    encoder.train()
    context_encoder.train()
    attack_clf.train()
    return embedding, encoder, context_encoder, attack_clf, voc


"""Compute training Iterations"""
def computerIterations(train_pairs):
    n_iter_per_epoch = len(train_pairs) // batch_size + int(len(train_pairs) % batch_size == 1)
    n_iteration = n_iter_per_epoch * finetune_epochs
    return n_iter_per_epoch, n_iteration

"""Create Optimizers and schedulers for training"""
def createOptimizer(models, type='adam'):
    if type == 'adam':
        optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
    elif type == 'sgd': 
        optimizer = torch.optim.SGD(models.parameters(), lr=learning_rate, momentum=0.9)
    opt_and_sched = OptimizerWithScheduler(models=models, optimizer=optimizer)
    return opt_and_sched

"""train"""




"""*** MODIFY patience and factor if needed**"""
def load_corpus_objects(utterance_path, conversation_path):
    """
    Load utterance and conversation dataframes from disk.
    """
    utt_df = pd.read_csv(utterance_path)
    conv_df = pd.read_csv(conversation_path)
    return utt_df, conv_df

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
    - add command line interface to set flags to run an experiment confirguation with:
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
    print("main")


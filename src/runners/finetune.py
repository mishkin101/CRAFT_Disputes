from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


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

"""Main inference piepeline implementing CRAFT. Because it uses the parent nn.mModule class, we can call .train() or .eval() on
    the entire model pipeline.
"""
class CraftPipeline(nn.Module):
    """This helper module encapsulates the CRAFT pipeline, defining the logic of passing an input through each consecutive sub-module."""
    def __init__(self, encoder, context_encoder, classifier, voc, optimizer, predictor, mode_flag):
        super(CraftPipeline, self).__init__()
        self.voc = voc
        self.encoder =encoder
        self.context_encoder = context_encoder
        self.classifier = classifier
        self.predictor = predictor
        self.optimizer = optimizer
        self.mode= mode_flag

    def forward(self, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels, val_score):
        _, utt_encoder_hidden = self.encoder(input_batch, utt_lengths)
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices)
        context_encoder_outputs, context_encoder_hidden = self.context_encoder(context_encoder_input, dialog_lengths)
        logits = self.classifier(context_encoder_outputs, dialog_lengths)
        if self.mode == 'predict':
            if self.predictor is None:
                    raise RuntimeError("predictor must be set in predict mode")
            """returns predicitons and scores"""
            return self.predictor(logits)
        elif self.mode == 'train':
            if labels is None:
                    raise RuntimeError("labels must be provided in training mode")
            loss = self.optimizer.calcLoss(logits, labels)
            self.opt.batchStep(logits, labels)
            return loss.item()
        elif self.mode == 'eval':
            if val_score is None:
                raise RuntimeError("validation score must be provided in evaluation mode")
            self.optimizer.epochStep(val_score)
            


    # def setmode(self):
    #     if self.predict_mode:
    #         self.encoder.train()
    #         self.context_encoder.train()
    #         self.attack_clf.train()
    #     else:
    #         self.encoder.eval()
    #         self.context_encoder.eval()
    #         self.attack_clf.eval() 


    
"""Load Device"""
def loadDevice():
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

"""Load Pretrained Model"""
def loadPretrainedModel():
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
def loadCheckpointandMode(checkpoint, device, classifier_type = 'single_target', mode = 'train'):
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
    toDevice([encoder,context_encoder, attack_clf])
    return embedding, encoder, context_encoder, attack_clf, voc

"""Convert Tensor to Device"""
def toDevice(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(device)
        elif isinstance(tensor, (list, tuple)):
             for t in tensor:
                t = toDevice(t, device)
        elif isinstance(tensor, dict):
            for v in tensor.values():
                  v  = toDevice(v, device) 
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}")



"""Compute training Iterations"""
def computerIterations(train_pairs):
    n_iter_per_epoch = len(train_pairs) // batch_size + int(len(train_pairs) % batch_size == 1)
    n_iteration = n_iter_per_epoch * finetune_epochs
    return n_iter_per_epoch, n_iteration

"""Create Optimizers and schedulers for training"""
def setOptimizer(models):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd': 
        optimizer = torch.optim.SGD(models.parameters(), lr=learning_rate, momentum=0.9)
    opt_and_sched = OptimizerWithScheduler(models=models, optimizer=optimizer)
    return opt_and_sched

"""Create loss funciton for training batch"""
def setLossFunction():
    if loss_function == 'bce':
         return nn.BCEWithLogitsLoss()

"""Handle logic for saving CRAFT model"""
def saveModel(craft_model):
    return

"""
Training Harness
Parameters:
    M: int, number of utterances in the batch   
    input_variable: tensorized input utterances for all contexts in a batch: (max_tokenized_length, M)
    dialog_lengths: tensor of dialog lengths for each utterance in the batch:  (batch_size, )
    dialog_lengths_list: list of dialog lengths for each utterance in the batch: [dialog_length_1, dialog_length_2, ...]
    utt_lengths: tensor of utterance lengths for each utterance in the batch: (M, )
    batch_indices: tensor of batch indices for each utterance in the batch: (M, )
    dialog_indices: tensor of dialog indices for each utterance in the batch: (M, )
    labels: tensor of labels for each context in the batch: (max_tokenized_length, batch_size)
"""

def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels, # input/output arguments
          encoder, context_encoder, attack_clf,                                                                   # optimization arguments
          optimizerCalc, lossCalc):                                                                                 # network arguments      
    #set to device options
    toDevice([input_variable, dialog_lengths, utt_lengths, labels])
    #First, we run the utterance tensors called (max_tokenized_length, M) to create utterance contexts
    """ Data: (max_tokenized_length, M, 500) -> X(max_tokenized_length, M, 500)
        Hidden: (2*2, M, 500)                  -> (2*2, M, 500) """
    _, utt_encoder_hidden = encoder(input_variable, utt_lengths)
    #Second, regroup the utterance's hidden states  into their respective dialogues
    """ Data:   (2*2, M, 500)                  -> X(max_convo_length, batch_size, 500)"""
    context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices)
    #Third, create the context encodings from the utterance states
    """ Data:   (max_dial_length, batch_size, 500)    -> (max_convo_length, batch_size, 500)
        Hidden: (2*1, M, 500)                         -> X(2*1, M, 500) """
    context_encoder_outputs, _ = context_encoder(context_encoder_input, dialog_lengths) 
    #make a pass through the classifier to get final logits for each conversation
    """(max_convo_length, batch_size, 500) -> ( batch_size, 1)"""
    logits = attack_clf(context_encoder_outputs, dialog_lengths)
    #update the loss each training iteration
    loss = lossCalc(logits,labels)
    optimizerCalc.batchStep(loss)
    return loss.item()


def evaluateBatch(craft_model, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices):
    # Set device options
    toDevice([input_batch, dialog_lengths, utt_lengths])
    # Predict future attack using predictor. return
    '''handle predictions and scores here'''
    return craft_model(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices)

def trainFinal(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels, craft_model):                                                                                           # optimization arguments): 
    toDevice([input_variable, dialog_lengths, utt_lengths, labels])
    return craft_model(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,dialog_indices)

"""Make predicitons on validation set from all batches"""
def validate(val_pairs, craft_model):
    #invoke iterator to  all needed artifacts for tensot converstion. No need to shuffle 
    batch_iterator = batchIterator(voc, val_pairs, batch_size, shuffle=False)
    n_iters = len(val_pairs) // batch_size + int(len(val_pairs) % batch_size > 0)
    all_preds = []
    all_labels = []
    for iteration in range(1, n_iters+1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        predictions, scores = evaluateBatch(craft_model, input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices)
    all_preds += [p.item() for p in predictions]
    all_labels += [l.item() for l in labels]
    print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))  
    return (np.asarray(all_preds) == np.asarray(all_labels)).mean()                                                        




"""*** MODIFY patience and factor if needed**"""
def load_corpus_objects():
    return
  


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


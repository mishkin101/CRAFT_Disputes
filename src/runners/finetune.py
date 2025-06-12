from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from typing import Type
from sklearn.metrics import get_scorer_names, get_scorer
import torch.nn.functional as F
from functools import partial


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
#import data utilties
from utils.data_processing import DataProcesser
#import corpys utilities
from utils.corpus_utils import *


# DEFAULT_CONFIG = {
#     "dropout": 0.1,
#     "batch_size": 64,
#     "clip": 50.0,
#     "learning_rate": 1e-5,
#     "print_every": 10,
#     "finetune_epochs": 30,
#     "validation_size": 0.2,
# }



"""Main inference piepeline implementing CRAFT. Because it uses the parent nn.mModule class, we can call .train() or .eval() on
    the entire model pipeline."""
class CraftPipeline(nn.Module):
    """This helper module encapsulates the CRAFT pipeline, defining the logic of passing an input through each consecutive sub-module."""
    def __init__(self, encoder, context_encoder, classifier, voc, optimizer, predictor, loss_function):
        super(CraftPipeline, self).__init__()
        self.voc = voc
        self.encoder =encoder
        self.context_encoder = context_encoder
        self.classifier = classifier #another nn.module potentially
        self.predictor = predictor #activation function
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.embedding = self.encoder.embedding if hasattr(self.encoder, "embedding") else None


    def forward(self, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels, epoch):
        _, utt_encoder_hidden = self.encoder(input_batch, utt_lengths)
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices)
        context_encoder_outputs, _= self.context_encoder(context_encoder_input, dialog_lengths)
        logits = self.classifier(context_encoder_outputs, dialog_lengths)
        if self.training:
            if labels is None:
                    raise RuntimeError("labels must be provided in training mode")
            loss = self.loss_function(logits, labels)
            self.optimizer.batchStep(logits, labels)
            return loss.item()
        else:
            if self.predictor is None:
                    raise RuntimeError("predictor must be set in eval mode")
            return self.predictor(logits)
""" 
Get the corpus object from chosen directory
If train mode: then return utterances and convo dataframe and perform context selection
"""
def loadDataset():
    data = DataProcesser(filepath=fine_raw_dir)
    contextSelection(data)
    return corpusBuilder(data)

"""
Which utterance to exlcude as context from meta.text
use functions defined in dataprocessor.py to create necessary dataframes
"""
def contextSelection(data: Type[DataProcesser]):
    filtered_df = data.filterRows("message", exclude_val= finetune_exclude_phrases, case_ex = finetune_case)
    data.setUtterancesDF(filtered_df)

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
def setClassifierHead():
    if classifier_type == 'single_target':
        single_clf = SingleTargetClf(hidden_size, dropout)
        return single_clf
    raise ValueError(f"Unsupported classifier type: {class_type}")


"""Build Contect encoder, decoder, and classifier"""
def loadCheckpoint(checkpoint):
    # Instantiate your modules
    voc             = loadPrecomputedVoc(corpus_name, word2index_path, index2word_path)
    attack_clf      = setClassifierHead()
    embedding       = nn.Embedding(voc.num_words, hidden_size)
    encoder         = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
    # Load weights
    embedding.load_state_dict(checkpoint['embedding'])
    encoder.load_state_dict(checkpoint['en'])
    context_encoder.load_state_dict(checkpoint['ctx'])
    if "atk_clf" in checkpoint:
        attack_clf.load_state_dict(checkpoint["atk_clf"])
    voc.__dict__ = checkpoint['voc_dict']
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
    # n_iteration = n_iter_per_epoch * finetune_epochs
    return n_iter_per_epoch#, n_iteration

"""Create Optimizers and schedulers for training"""
"""Models: models[0]: encoder, models[1]: context_encoder, models[2]:attack_clf"""
def setOptimizer(models):
    models = nn.ModuleList([models[0], models[1], models[2]])
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd': 
        optimizer = torch.optim.SGD(models.parameters(), lr=learning_rate, momentum=0.9)
    opt_and_sched = OptimizerWithScheduler(models=models, optimizer=optimizer)
    return opt_and_sched

"""Create loss funciton for training batch from nn.modules.loss"""
def setLossFunction():
    candidates = [loss_function, loss_function[0].upper() + loss_function[1:]]
    for name in candidates:
        if hasattr(nn, name):
            LossClass = getattr(nn, name)
            if isinstance(LossClass, type) and issubclass(LossClass, nn.Module):
                return LossClass()
        else:
            raise ValueError(f"`{loss_function}` is not a callable or nn.Module subclass")

"""Create activation function for Predictor Module from nn.functional"""
def setPredictorActivation(**kwargs):
    name = activation.lower()
    if not hasattr(F, name):
        raise ValueError(f"`{activation}` is not a valid torch.nn.functional activation")
    func = getattr(F, name)
    # Wrap in partial to bind kwargs (e.g. dim for softmax)
    return partial(func, **kwargs)
    
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

"""return thresholded score logic"""
def classificationThreshold(scores):
    return (scores > forecast_thresh).float()

"""return validation score metric"""
def valScore(preds, labels):
    scorer_obj = get_scorer(score_function)
    metric_fn   = scorer_obj._score_func
    return metric_fn(labels, preds)

"""return scores from predictor"""
def evaluateBatch(craft_model, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices):
    toDevice([input_batch, dialog_lengths, utt_lengths])
    return craft_model(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices)

"""return loss from CRAFT pipeline and update optimzer steps, gradients"""
def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels, craft_model):                                                                                           # optimization arguments): 
    toDevice([input_variable, dialog_lengths, utt_lengths, labels])
    return craft_model(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,dialog_indices)

"""Make predicitons on validation set or test set from all batches"""
def evaluate(pairs, craft_model):
    results = {}
    #invoke iterator to  all needed artifacts for tensot converstion. No need to shuffle 
    batch_iterator = batchIterator(voc, pairs, batch_size, shuffle=False)
    n_iters = len(pairs) // batch_size + int(len(pairs) % batch_size > 0)
    with torch.no_grad():
        for iteration in range(1, n_iters+1):
            batch, batch_dialogs, *_ = next(batch_iterator)
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, id_batch,*_ = batch
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            scores = evaluateBatch(craft_model, input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices)
            predictions = classificationThreshold(scores)
        for i, comment_id in enumerate(id_batch):
                results[comment_id] = {
                    "probability": scores[i].detach().cpu().item(),
                    "prediction": predictions[i].detach().cpu().item(),
                    "label": labels[i].detach().cpu().item() 
                }
    return results                                                

def trainIter(train_pairs, val_pairs, craft_model, epoch_iterations):
    best_val_score = 0
    train_history = []
    val_history = []
    best_model_path =None
    for epoch in range(finetune_epochs):
        total_loss =0
        for iteration in range(1, epoch_iterations +1):
            training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
            batch_iterator = batchIterator(voc, train_pairs, batch_size)
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, *_ = training_batch
            dialog_lengths_list = [len(x) for x in training_dialogs]
            loss = train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels, craft_model)
            total_loss += loss
        craft_model.eval()
        results = evaluate(val_pairs, craft_model)
        all_preds  = [ entry["prediction"] for entry in results.values() ]
        all_labels = [ entry["label"]      for entry in results.values() ]
        val_score = valScore(all_preds, all_labels)
        if val_score > best_val_score:
            best_val_score = val_score
            saveModel(craft_model, loss, epoch)
            best_model_path = os.path.join(experiment_model_dir, f"{experiment_name}_best_epoch_{epoch}.tar") 
        craft_model.optimizer.epochStep(val_score)
        train_history.append(total_loss)
        val_history.append(val_score)
        print(f"[Epoch {epoch}/{finetune_epochs}] train_loss={loss:.4f}  val_score={val_score:.4f}")
        craft_model.train()
    return train_history, val_history, best_model_path




def main(config):
    globals().update(config)
    #handle logic for loading data:
    utterance_metadata = finetune_utterance_metadata 
    conversation_metadata = finetune_convo_metadata
    loaded_corpus = loadDataset()
    #get conversations and utterances dataframe:
    convo_dataframe = loaded_corpus.get_conversations_dataframe()
    utterance_dataframe = loaded_corpus.get_utterances_dataframe()
    #load device:
    device = loadDevice()
    #handle loading pre-trained model:
    model_path = os.path.join(save_dir_pretrain, pretrain_model)
    pretrained_checkpoint = torch.load(f= model_path, map_location = device)
    pretrain_model = loadPretrainedModel(pretrained_checkpoint):
    #handle loading model artifacts:
    embedding, encoder, context_encoder, attack_clf, voc = loadCheckpoint(checkpoint)
    #load optimzer:
    models = [encoder, context_encoder, attack_clf]
    optim = setOptimizer(models)
    loss_fn = setLossFunction()
    #create predictor that stores activation function:
    activation_fn = setPredictorActivation()
    predictor = setPredictor(activation_fn)
    #create CRAFT Pipeline:
    craft = CraftPipeline(encoder, context_encoder, attack_clf, voc, optim, predictor, loss_fn)
    #create training logic:
    





    return



""" 
1.Data Pre-processing: 

    processDialog() *** DONE ***
        -  tokenization of texts in eatch batch 

    loadPairs() *** DONE ***
        -   loading pairs according to the split they are in as (context, reply, label, comment_id). 
        -   Called by processDialog()

    contextSelection():  *** DONE ***
        - which utterance to exlcude as context from meta.text

2.Loading Corpus Objects: 
    loadCorpusObjects() *** DONE ***
        -  load the corpus objects from corpus_dir
        -  load the utterances, speakers, and conversations dataframes
        -  this will be used to get the utterances and their metadata for training and evaluation
    load all corpora into memory

3.Load Vocabulary Object: *** DONE ***
    loadPrecomputedVoc()

4.Build Classifiers:  
    NEW Predictor(nn.Module): - Subjecitve outcomes *** NEED TO MAKE ***
    Predictor(nn.Module): - Objective outcomes (single target) Exists in model.py *** DONE ***

5. Build Predictor Harness:(used in test/eval tests) *** DONE ***
    - runs through utt encoder + contetext encoder to create hidden states
    - calls classifier to get predicitons from encoded context

6. Build Train Loop: (for fine-tuning training) *** DONE ***
     src.train()
    - zero out gradients on every mini-batch
    - runs through utt encoder + contetext encoder to create hidden states
    - calculate loss
    - clip gradients to prevent exploding
    - adjust weights
    - return loss

7. Build evaluateBatch: *** DONE ***
    evaluateBatch() 
    - returns the (predicitons) and scores for test/val tests

8. Build evaluation logig for validation and predicition *** DONE ***
    evaluate() 

12. Handle K-fold splitting in training *** NEED TO MAKE ***
    
9. Managing saving experiment results  *** NEED TO MAKE ***
    src.file_utils.save_experiment_results()
    - save model state dict
    - save training history
    - save validation history
    - save test history
    - save hyper-parameters used in the experiment
    - save model architecture (if changed)
    - save final utterances, conversations dataframes to experiment in "experiments" directory

10. Plotting Utililties *** NEED TO MAKE ***
 *** FIND GOOD PACKAGE TO MANAGE MONITORING TRAINING ***
    - plot training history
    - plot validation history
    - plot test history
    - plot hyper-parameters used in the experiment
    - save plots to "experiments" directory

12. build hyper-parameter tuning with parallelization *** NEED TO MAKE ***
 *** FIND GOOD PACKAGE TO MANAGE MONITORING TRAINING ***
    - use Optuna to tune hyper-parameters and parallelize
    - save best hyper-parameters to "experiments" directory

13. us MLFlow to manage the training pipeline *** NEED TO MAKE ***
    - use MLFow to track experiments, hyper-parameters, and results
    - save model artifacts to MLFow server
    - log training history, validation history, and test history to MLFow server
 *** FIND GOOD PACKAGE TO MANAGE MONITORING TRAINING ***
    - use Optuna to tune hyper-parameters and parallelize
    - save best hyper-parameters to "experiments" directory

11. Build and create Fine-tuning harness: *** NEED TO MAKE ***
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
        - what metadata to include in conversations
            - modify the conversation_ metadata in corpus_utils
            - modify the utterances_detadata in corpus_utils


"""

if __name__ == "__main__":
    voc = loadPrecomputedVoc(corpus_name, word2index_path, index2word_path)

    



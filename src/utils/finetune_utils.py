from functools import partial
from collections import defaultdict
from typing import Type
from typing import Dict, List
from sklearn.metrics import get_scorer
from ray.tune import tune
from torch.nn import functional as F
import shutil
import mlflow




# import all configuration variables
from model.config import *
import model.config as cfg_mod
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
#import file utils
from utils.file_utils import *
#importing plotting functions:
from utils.plotting_utils import *




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


    def forward(self, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels):
        _, utt_encoder_hidden = self.encoder(input_batch, utt_lengths)
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices)
        context_encoder_outputs, _= self.context_encoder(context_encoder_input, dialog_lengths)
        logits = self.classifier(context_encoder_outputs, dialog_lengths)
        if self.training:
            if labels is None:
                    raise RuntimeError("labels must be provided in training mode")
            loss = self.loss_function(logits, labels)
            self.optimizer.batchStep(loss)
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
    # if data:
    #     fine_raw_file = data[0]
    print(f"getting data from {fine_raw_file}")
    print(f"dropping conversation values: {finetune_conversation_map}")
    print(f"dropping utterance values: {finetune_utterance_map}")
    print(f"including AI?: {finetune_include_AI}")
    data = DataProcesser(filepath=fine_raw_file)
    data.contextSelection(finetune_conversation_map, finetune_utterance_map)
    exp_processed_file = os.path.join(experiment_dir, fine_processed_filename)
    # data.saveToCSV(exp_processed_file, drop_parsed=True)
    print("displaying processes dataframes:")
    display(data.getDataframe())
    print("displaying utterance dataframe:")
    display(data.getUtterancesDF())
    return corpusBuilder(data)

# def saveDataset(data):
#     save_loc = os.path.join(fine_processed_dir, fine_processed_file)
#     print(f"saving processed corpus to:")

# """
# Which utterance to exlcude as context from meta.text
# use functions defined in dataprocessor.py to create necessary dataframes
# """
# def contextSelection(data: Type[DataProcesser]):
#     filtered_df = data.filterRows("message", exclude_val= finetune_exclude_phrases, case_ex = finetune_case)
#     data.setUtterancesDF(filtered_df)

"""Load Device"""
def loadDevice():
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

"""Load Pretrained Model checkpoint"""
def loadPretrainedModel():
    if not os.path.exists(pretrain_model_path):
        raise FileNotFoundError(f"Model file not found at {pretrain_model_path}")
    if device == 'cuda':
        print("Loading model on GPU")
        checkpoint = torch.load(pretrain_model_path)
    else:
        print("Loading model on CPU")
        print(f"loading pretrained model from {pretrain_model_path}")
        checkpoint = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
    return checkpoint

"""Create Classifier Head"""
def setClassifierHead():
    if classifier_type == 'single_target':
        single_clf = SingleTargetClf(hidden_size, dropout)
        return single_clf
    raise ValueError(f"Unsupported classifier type")


"""Build Contect encoder, decoder, and classifier"""
def loadCheckpoint(checkpoint):
    # Instantiate your modules
    print(f"Loading vocab from {word2index_path}")
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
                t = toDevice(t)
        elif isinstance(tensor, dict):
            for v in tensor.values():
                  v  = toDevice(v) 
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}")
        

"""Compute training Iterations"""
def computerIterations(train_pairs):
    n_iter_per_epoch = len(train_pairs) // batch_size + int(len(train_pairs) % batch_size != 0)
    print(f"the number of train pairs is:{len(train_pairs)}")
    print(f"computed n_iter_per epoch is: {n_iter_per_epoch}")
    # n_iteration = n_iter_per_epoch * finetune_epochs
    return n_iter_per_epoch



"""Create Optimizers and schedulers for training"""
"""Models: models[0]: encoder, models[1]: context_encoder, models[2]:attack_clf"""
def setOptimizer(models):
    try:
        models = nn.ModuleList([models[0], models[1], models[2]])
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
        if optimizer_type == 'sgd': 
            optimizer = torch.optim.SGD(models.parameters(), lr=learning_rate, momentum=0.9)
        print(f"Loading {optimizer_type} optimizer  with scheduler setting: {scheduling}")
    except:
         raise ValueError(f"`{optimizer_type}` is not a callable or torch.optim subclass")
    try:
        opt_and_sched = OptimizerWithScheduler(models=models, optimizer=optimizer)
        return opt_and_sched
    except:
        raise ValueError(f"scheduler could not be configured")


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

"""return validation score metrics"""
def valScore(preds, labels, probs):
    results = {}
    # Metrics that require probability inputs
    prob_based = {"neg_brier_score", "neg_log_loss", "roc_auc", "average_precision"}
    for score_fn in score_functions:
        scorer_obj = get_scorer(score_fn)
        metric_fn   = scorer_obj._score_func
        if score_fn in prob_based:
            results[score_fn] = metric_fn(labels, probs)
        else:
            results[score_fn] = metric_fn(labels, preds)
    return results

"""return scores from predictor"""
def evaluateBatch(craft_model, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels):
    # print(f"Loaded eval tensors to device")
    toDevice([input_batch, dialog_lengths, utt_lengths])
    return craft_model(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels)

"""return loss from CRAFT pipeline and update optimzer steps, gradients"""
def train(craft_model,input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels):
    # print(f"Loaded train tensors to device")                                                                                          # optimization arguments): 
    toDevice([input_variable, dialog_lengths, utt_lengths, labels])
    return craft_model(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, labels)

"""Make predicitons on validation set or test set from all batches"""
def evaluate(voc, pairs, craft_model):
    results = {}
    #invoke iterator to  all needed artifacts for tensor conversion. No need to shuffle 
    batch_iterator = batchIterator(voc, pairs, batch_size, shuffle=False)
    n_iters = len(pairs) // batch_size + int(len(pairs) % batch_size > 0)
    with torch.no_grad():
        for iteration in range(1, n_iters+1):
            batch, batch_dialogs, batch_labels, true_batch_size = next(batch_iterator)
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, id_batch,*_ = batch
            print(f"the size of this batch is:{true_batch_size}")
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            scores = evaluateBatch(craft_model, input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, true_batch_size, batch_labels)
            predictions = classificationThreshold(scores)
            for i, comment_id in enumerate(id_batch):
                    results[comment_id] = {
                        "probability": scores[i].detach().cpu().item(),
                        "prediction": predictions[i].detach().cpu().item(),
                        "label": labels[i].detach().cpu().item() 
                    }
    return results                                                



def trainEpoch(train_pairs, craft_model, epoch_iterations, voc, total_loss, epoch):
    print(f"starting training epoch {epoch}...")
    batch_losses = []
    label_counts = []
    # print_every = max(1, epoch_iterations // 10)
    batch_iterator = batchIterator(voc, train_pairs, batch_size)
    craft_model.train()
    print(f"training iterations per epoch: {epoch_iterations}")
    for iteration in range(0, epoch_iterations):
        training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, *_ = training_batch
        dialog_lengths_list = [len(x) for x in training_dialogs]
        loss = train(craft_model, input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, true_batch_size, labels)
        total_loss += loss
        print(f"the true batch size is:{true_batch_size}")
        if iteration % print_every == 0:
            batch_losses.append({"epoch": epoch, "iteration": (epoch-1)*epoch_iterations + iteration + 1,"loss": loss})
            print("train loss is:", loss)
        flat = labels.view(-1).long()
        counts = torch.bincount(flat).cpu().numpy()
        label_counts.append(counts)
    print(f"finished training with total loss: {total_loss}")
    return batch_losses, label_counts

def evalEpoch(voc, val_pairs, craft_model, epoch):
        print(f"starting eval epoch {epoch}...")
        craft_model.eval()
        results = evaluate(voc, val_pairs, craft_model)
        all_preds  = [ entry["prediction"] for entry in results.values() ]
        all_labels = [ entry["label"]      for entry in results.values() ]
        all_probs  = [ entry["probability"] for entry in results.values() ]
        print(results)
        val_scores = valScore(all_preds, all_labels, all_probs)
        craft_model.optimizer.epochStep(val_scores[epoch_scheduling_metric])
        print(f"finished eval for epoch {epoch} with val accuracy:", val_scores["accuracy"])
        return {"epoch": epoch, "val_scores": val_scores}

def average_across_folds(all_folds):
    n_folds = len(all_folds)
    n_epochs = len(all_folds[0]) 
    mean_per_epoch = []
    for epoch in range(1, n_epochs+1):
        sums = defaultdict(float)
        for fold in all_folds:
            scores = fold[epoch-1]["val_scores"]
            for metric, value in scores.items():
                sums[metric] += value
        means = {metric: sums[metric] / n_folds for metric in sums}
        mean_per_epoch.append({"epoch": epoch, "mean_val_scores": means})
    return mean_per_epoch




"""Handle loading fresh model for every fold"""
def loadModelArtifacts():
    print(f"Loading model artifacts...")
    #load device:
    device = loadDevice()
    #handle loading pre-trained model:
    pretrained_checkpoint = loadPretrainedModel()
    #handle loading pre-trained model artifacts:
    embedding, encoder, context_encoder, attack_clf, voc = loadCheckpoint(pretrained_checkpoint)
    #load optimzer:
    models = [encoder, context_encoder, attack_clf]
    optim = setOptimizer(models)
    loss_fn = setLossFunction()
    #create predictor that stores activation function:
    activation_fn = setPredictorActivation()
    predictor = Predictor(activation_fn)
    #create CRAFT Pipeline:
    craft_model = CraftPipeline(encoder, context_encoder, attack_clf, voc, optim, predictor, loss_fn)
    craft_model.to(device)
    return craft_model, voc, optim

"""handle loading data"""
def loadDataArtifacts():
    print(f"Loading data artifacts...")
    #handle logic for loading data:
    loaded_corpus = loadDataset()
    #get conversations and utterances dataframe:
    convo_dataframe = loaded_corpus.get_conversations_dataframe()
    utterance_dataframe = loaded_corpus.get_utterances_dataframe()
    return convo_dataframe, utterance_dataframe

def finetune_craft(config):
    # try:
        """=== MLFLOW ==="""
        mlflow.set_tracking_uri("http://127.0.0.1:8080")        # HTTP server URI :contentReference[oaicite:2]{index=2}
        mlflow.set_experiment(config['experiment_name']) 
        with mlflow.start_run():
            apply_config(config)
            mlflow.log_params(config)
            convo_dataframe, utterance_dataframe = loadDataArtifacts()
            #create training logic:
            X_train_id, X_test_id, y_train_id, y_test_id = createTrainTestSplit(convo_dataframe)
            convo_dataframe_main = assignSplit(convo_dataframe, train_ids=X_train_id, test_ids=X_test_id)
            X_train = convo_dataframe.loc[X_train_id]
            X_test = convo_dataframe.loc[X_test_id]
            #same splits for each k-fold index
            train_val_id_list = createTrainValSplit(X_train)
            fold_models = []
            fold_dataframes = []
            fold_opts = []
            fold_data = []
            fold_batch_metrics = {f"fold_{i}": [] for i in range(1, k_folds+1)}  # will hold per‐batch dicts
            fold_epoch_metrics    = {f"fold_{i}": [] for i in range(1, k_folds+1)}  # will hold per‐epoch dicts
            fold_train_total_loss = {i: 0.0 for i in range(1, k_folds + 1)}
            # load model for each fold:
            for fold, pair in enumerate(train_val_id_list, start=1):
                print(f"=== Loading fold artifacts for fold {fold} ===")
                craft_model, voc, optim = loadModelArtifacts()
                print(f"Loading fold directories")
                build_fold_directories(fold)
                print(f"Loading train/val pairs")
                convo_dataframe_fold = assignSplit(convo_dataframe, train_ids=pair[0], val_ids=pair[1])
                train_pairs = loadLabeledPairs(voc, utterance_dataframe, convo_dataframe_fold, last_only = last_only_train, split_key="train")
                val_pairs = loadLabeledPairs(voc, utterance_dataframe, convo_dataframe_fold, last_only = last_only_val, split_key="val")
                fold_models.append(craft_model)
                fold_opts.append(optim)
                fold_data.append((train_pairs, val_pairs))
                fold_dataframes.append(convo_dataframe_fold)

            """=== TESTING PLOTTING ==="""
            fig,_ = plot_fold_summary_with_ai(
                fold_dataframes,
                split_col="meta.split",
                outcome_col="meta.provided_outcome",
                length_col="meta.convo_len",
                preferred_splits=("train","val")
            )
            mlflow.log_figure(fig, "fold_summary.png")
            plt.close(fig)

            """Can maybe parallelize this to have all folds running one epoch at same time """
            # train each fold per epoch to implement early stopping with avg-val-score of choice
            for epoch in range(1, finetune_epochs + 1):
                for i in range(k_folds):
                    model_i = fold_models[i]
                    train_pairs, val_pairs = fold_data[i]
                    #create epoch iterations:
                    epoch_iters = computerIterations(train_pairs)
                    #{"batch_losses": {"epoch": epoch, "iteration": iteration,"loss": loss}}
                    batch_metrics, label_counts = trainEpoch(train_pairs, model_i, epoch_iters, voc, fold_train_total_loss[i+1], epoch)
                    print(f"label counts for all batches: \n {label_counts}")
                    """=== MLFLOW= ==="""
                    for bm in batch_metrics:
                        mlflow.log_metric(
                            key=f"fold_{i+1}_train_loss",
                            value=bm["loss"],
                            step=(bm["epoch"] - 1) * epoch_iters + bm["iteration"]
                        )
                    fig = plot_batch_distributions(label_counts,fold, epoch, label_map)
                    mlflow.log_figure(fig, artifact_file = f"fold_{i}/epoch_{epoch:02d}_batch_label_distribution.png")
                    plt.close()
                    #{"epoch": epoch, "val_scores": val_scores}
                    #val_scores = {"score":val, ...}
                    epoch_metrics = evalEpoch(voc, val_pairs, model_i, epoch)
                    """=== MLFLOW= =="""
                    for score_fn, val in epoch_metrics["val_scores"].items():
                        mlflow.log_metric(
                            key=f"fold_{i+1}_val_{score_fn}",
                            value=val,
                            step=epoch
                        )
                    fold_batch_metrics[f"fold_{i+1}"].append(batch_metrics)
                    fold_epoch_metrics[f"fold_{i+1}"].append(epoch_metrics)
                all_folds = list(fold_epoch_metrics.values())
                #{"epoch": epoch, "val_scores": val_scores}
                mean_per_epochs = average_across_folds(all_folds)
                mean_scores_this_epoch = mean_per_epochs[-1]["mean_val_scores"]
                """=== MLFLOW= =="""
                for score_fn, mean_val in mean_scores_this_epoch.items():
                    mlflow.log_metric(
                        key=f"mean_val_{score_fn}",
                        value=mean_val,
                        step=epoch
                    )
                if ray_tune:
                    all_metrics = {**log_fold_to_tune(epoch, fold_batch_metrics), **log_epoch_to_tune(epoch, mean_per_epochs)}
                    tune.report(all_metrics)
                else:
                    print(f"Logging metrics for epoch {epoch}")
                    for fold_idx in range(1, k_folds+1):
                        log_folds(fold_idx, "training", "epoch_metrics.txt", fold_epoch_metrics[f"fold_{fold_idx}"][-1])
                        log_folds(fold_idx, "training", "batch_metrics.txt", fold_batch_metrics[f"fold_{fold_idx}"][-1])
                    log_exp("training", "avg_metrics.txt", mean_per_epochs[-1])
            if not ray_tune:
                log_exp("config", "config.txt", config)
            """=== MLFLOW= =="""

        # mlflow.end_run()
    # except Exception as e:
    #     print(f"Experiment failed: {e!r}")
    #     print(f"Cleaning up {experiment_dir!r}...")
    #     shutil.rmtree(experiment_dir, ignore_errors=True)
    # else:
    #     print(f"Finetuning successfully completed. Training results stored in {experiment_dir}.")
        


    
def log_fold_to_tune(epoch: int, fold_batch_metrics):
    report_dict = {}
    for fold_name, epoch_list in fold_batch_metrics.items():
        #add another epoch level here
        for epoc in epoch_list:
            for batch in epoc:
                if batch["epoch"] != epoch:
                    continue
                iteration = batch["iteration"]
                loss = batch["loss"]
                report_dict[f"{fold_name}_train_loss_{iteration}"] = loss
    return report_dict

def log_epoch_to_tune(epoch: int, mean_per_epochs):
    report_dict = {}
    mean_entry = mean_per_epochs[epoch-1]["mean_val_scores"]
    for metric_name, metric_val in mean_entry.items():
        report_dict[f"mean_val_{metric_name}"] = metric_val
    return report_dict

def apply_config(config):
    for key, val in config.items():
        if hasattr(cfg_mod, key):
            setattr(cfg_mod, key, val)
        else:
            continue



def finetune_craft_test(config):
            apply_config(config)
            convo_dataframe, utterance_dataframe = loadDataArtifacts()
            #create training logic:
            X_train_id, X_test_id, y_train_id, y_test_id = createTrainTestSplit(convo_dataframe)
            convo_dataframe_main = assignSplit(convo_dataframe, train_ids=X_train_id, test_ids=X_test_id)
            X_train = convo_dataframe.loc[X_train_id]
            X_test = convo_dataframe.loc[X_test_id]
            #same splits for each k-fold index
            train_val_id_list = createTrainValSplit(X_train)
            fold_models = []
            fold_dataframes = []
            fold_opts = []
            fold_data = []
            fold_batch_metrics = {f"fold_{i}": [] for i in range(1, k_folds+1)}  # will hold per‐batch dicts
            fold_epoch_metrics    = {f"fold_{i}": [] for i in range(1, k_folds+1)}  # will hold per‐epoch dicts
            fold_train_total_loss = {i: 0.0 for i in range(1, k_folds + 1)}
            # load model for each fold:
            for fold, pair in enumerate(train_val_id_list, start=1):
                print(f"=== Loading fold artifacts for fold {fold} ===")
                craft_model, voc, optim = loadModelArtifacts()
                print(f"Loading fold directories")
                build_fold_directories(fold)
                print(f"Loading train/val pairs")
                convo_dataframe_fold = assignSplit(convo_dataframe, train_ids=pair[0], val_ids=pair[1])
                train_pairs = loadLabeledPairs(voc, utterance_dataframe, convo_dataframe_fold, last_only = last_only_train, split_key="train")
                val_pairs = loadLabeledPairs(voc, utterance_dataframe, convo_dataframe_fold, last_only = last_only_val, split_key="val")
                fold_models.append(craft_model)
                fold_opts.append(optim)
                fold_data.append((train_pairs, val_pairs))
                fold_dataframes.append(convo_dataframe_fold)
            """Can maybe parallelize this to have all folds running one epoch at same time """
            # train each fold per epoch to implement early stopping with avg-val-score of choice
            for epoch in range(1, finetune_epochs + 1):
                for i in range(k_folds):
                    model_i = fold_models[i]
                    train_pairs, val_pairs = fold_data[i]
                    #create epoch iterations:
                    epoch_iters = computerIterations(train_pairs)
                    #{"batch_losses": {"epoch": epoch, "iteration": iteration,"loss": loss}}
                    batch_metrics, label_counts = trainEpoch(train_pairs, model_i, epoch_iters, voc, fold_train_total_loss[i+1], epoch)
                    print(f"label counts for all batches: \n {label_counts}")
                    # fig = plot_batch_distributions(label_counts,fold, epoch, label_map)
                    # fig_path = f"fold_{i}/epoch_{epoch:02d}_batch_label_distribution.png"
                    # fig.savefig(fig_path)
                    # fig.clf()

                    #{"epoch": epoch, "val_scores": val_scores}
                    #val_scores = {"score":val, ...}
                    epoch_metrics = evalEpoch(voc, val_pairs, model_i, epoch)
                    fold_batch_metrics[f"fold_{i+1}"].append(batch_metrics)
                    fold_epoch_metrics[f"fold_{i+1}"].append(epoch_metrics)
                all_folds = list(fold_epoch_metrics.values())
                #{"epoch": epoch, "val_scores": val_scores}
                mean_per_epochs = average_across_folds(all_folds)
                mean_scores_this_epoch = mean_per_epochs[-1]["mean_val_scores"]
                print(f"fold_batch_metricsis \n{fold_batch_metrics}")
                if ray_tune:
                    all_metrics = {**log_fold_to_tune(epoch, fold_batch_metrics), **log_epoch_to_tune(epoch, mean_per_epochs)}
                    tune.report(all_metrics)
                else:
                    print(f"Logging metrics for epoch {epoch}")
                    for fold_idx in range(1, k_folds+1):
                        log_folds(fold_idx, "training", "epoch_metrics.txt", fold_epoch_metrics[f"fold_{fold_idx}"][-1])
                        log_folds(fold_idx, "training", "batch_metrics.txt", fold_batch_metrics[f"fold_{fold_idx}"][-1])
                    log_exp("training", "avg_metrics.txt", mean_per_epochs[-1])
            if not ray_tune:
                log_exp("config", "config.txt", config)
            """=== MLFLOW= =="""

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



    



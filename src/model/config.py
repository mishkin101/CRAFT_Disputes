import os.path
from pathlib import Path


# get the absolute path to the repository so we don't have to deal with relative paths
repo_dir = Path(__file__).parent.parent.absolute()

corpus_name = "custom" # Name of the dataset to run CRAFT on. This is not directly used by the model, it is instead used by
                         # this config file to define certain input and output locations. You can, of course, override those
                         # location settings directly and thus completely ignore this setting, it is just useful to use this
                         # setting to keep everything consistent and organized :)
                         # Note that in addition to the default setting of "wikiconv" you can also set this to "cmv" and still
                         # have the code work out-of-the-box (with Reddit CMV data) as the repo includes all support files needed
                         # for both the Wikiconv and Reddit CMV corpora.
# Name of the conversation metadata field in the ConvoKit corpus to use as the label for training and evaluation.
# Note that the if-statement in the default value is only there to enable users to switch to the CMV data by only
# changing the corpus_name (rather than having to directly change label_metadata as well). If you are using a custom
# corpus, then of course the if-statement is not needed and you can just directly put the name of the metadata

# field in your corpus that you want to use as the label.
if corpus_name == "wikiconv":
   label_metadata = "conversation_has_personal_attack" 
if corpus_name == "cmv":
   label_metadata = "has_personal_attack" 
if corpus_name == "custom":
   label_metadata = "provided_outcome"

label_map = {0:"success", 1:"impasse"}

# Name of the utterance metadata field that contains comment-level toxicity labels, if any. Note
# that CRAFT does not strictly need such labels, but some datasets like the wikiconv data do include
# it. For custom datasets it is fine to leave this as None.
if corpus_name == "wikiconv": 
   utt_label_metadata = "comment_has_personal_attack" 
if corpus_name == "cmv": 
   utt_label_metadata = None
if corpus_name == "custom":
   utt_label_metadata = None
# Name of the directory where the ConvoKit corpus objects will be saved. 
corpora = "corpora" 
# Name of the fine-tuning corpus to use for fine-tuning
finetune_corpus_name = "kodis"
# Name of the fine-tuning dataset to use for fine-tuning
finetune_data = "KODIS-EN.csv"
#Name of the pretrained model file:
pretrained_model = "model.tar"
# define file locations
data_dir = os.path.join(repo_dir, "data") # Where to save the pre-processed data files
save_dir_pretrain = os.path.join(repo_dir, "saved_models", corpus_name, "pretrained_models")
save_dir_finetune = os.path.join(repo_dir, "saved_models", corpus_name, "finetuned_models")
pretrain_model_path = os.path.join(save_dir_pretrain, pretrained_model)

# os.path.join(repo_dir, "saved_models", corpus_name) # Where to save the pre-trained model
corpus_dir = os.path.join(data_dir, corpora) # Where to save the ConvoKit corpus object
train_path = os.path.join(data_dir, "nn_input_data", corpus_name, "train_processed_dialogs.txt") # File containing unlabeled data for pre-training
word2index_path = os.path.join(data_dir, "nn_preprocessing", corpus_name, "word2index.json") # These two files jointly define the
index2word_path = os.path.join(data_dir, "nn_preprocessing", corpus_name, "index2word.json") # model's vocabulary 
experiments_dir = os.path.join(repo_dir, "experiments")

#saved directory for fine-tuning dataset:µ
fine_dir = os.path.join(data_dir,"finetuning_preprocessing")
fine_raw_dir = os.path.join(data_dir, "finetuning_preprocessing", "raw", finetune_corpus_name) # Where to save the raw data files
fine_processed_dir = os.path.join(data_dir, "finetuning_preprocessing", "processed", finetune_corpus_name) # Where to save the processed data files
fine_raw_file =  os.path.join(fine_raw_dir, finetune_data)

#set processed file on each  finetune run
fine_processed_filename = "change_to_custom.csv"
fine_processed_file = os.path.join(fine_processed_dir, fine_processed_filename)

#metadata pretraining:
pretrain_include_AI = True
pretrain_utterance_metadata = []
pretrain_convo_metadata = []

#values to to exlude from fine-tuning:
#select custom conversation and utterance cols and corrsponding values with case if applicable.
#col, [values, case, include/exclude]

finetune_conversation_map = {
   'buyer_is_AI': {"include": [[False], None], "exclude": [[], None]},
   'seller_is_AI':{"include": [[False], None], "exclude": [[], None]}
}

finetune_utterance_map = {
   'message': {"include": [[], False], "exclude": [[], True]},
   'speaker_id': {"include": [[], None], "exclude": [[], None]},
   'is_AI':  {"include": [[], None], "exclude": [[], None]},
}

finetune_include_AI = False


#context selection for train and test:
last_only_train = True
last_only_val = True
last_only_test = False

utterance_headers = ["id", "speaker", "conversation_id", "reply_to", "timestamp", "text"]
utterance_metadata =  ["predictions", "scores"]
speaker_headers = ['id']
speaker_metadata = None#["b_country", "s_country", "is_AI"]
conversation_headers = ['id']  #["id", "name", "timestamp", "num_utterances"]
conversation_metadata = ["buyer_is_AI", "seller_is_AI", "convo_len", "provided_outcome", "s_SVI_instrumental", "s_SVI_self", "s_SVI_process", "s_SVI_relationship"] #["num_turns", "dispute"]

# Configure model architecture parameters
attn_model = 'general'
MAX_LENGTH = 80  # Maximum sentence length to consider
CONTEXT_SIZE = 16 # Maximum conversational context length to consider
hidden_size = 500 # Hidden size of the utterance and context embeddings
encoder_n_layers = 2 # Number of layers in the utterance encoder
context_encoder_n_layers = 2 # Number of layers in the context encoder
decoder_n_layers = 2 # Number of layers in the decoder
dropout = 0.1 # Dropout rate
batch_size = 64 # Number of conversations per batch

# Configure training/optimization
pretrain_epochs = 3 if corpus_name == "wikiconv" else 6 # Number of pre-training epochs. Smaller by default for wikiconv since it has a larger training corpus
finetune_epochs = 30 # Number of fine-tuning epochs
clip = 50.0 # Maximum gradient cutoff during training
teacher_forcing_ratio = 1.0 # How often to use ground-truth instead of generated output when training the decoder during pre-training phase
learning_rate = 0.0001 # Learning rate to use during pre-training
labeled_learning_rate = 1e-5 # Learning rate to use during fine-tuning
decoder_learning_ratio = 5.0 # Learning rate multiplier on the decoder layers
print_every = 10 # How often to print output to the screen (measured in training iterations)
# forecast_thresh = 0.570617 if corpus_name == "wikiconv" else 0.548580 # CRAFT score above which the forecast is considered positive. The default values were tuned on validation data for each corpus.
forecast_thresh = .5

#Optimizer:
#Options: 'adam', 'sgd'
optimizer_type = 'adam'  
#Scheduler: ReduceLROnPlateau
#Options:
patience = 5
factor = .1
threshold = 0.0001
mode = 'max'  # Mode for ReduceLROnPlateau scheduler, 'min' for minimizing loss
scheduling = True
epoch_scheduling_metric = 'accuracy'

#Imbalance Strategy:
#Options: "none", "stratified", "downsampling, 
imbalance_handling = "none"

#Loss:
#Options: any loss function from nn.modules.loss. See "__all__" 
loss_function = 'BCEWithLogitsLoss'  
pos_weight = 1
#type of device
device = "cpu" 

#Number of Folds
k_folds = 3
#Epoch Score Functions
#Options: any score metric name from sklearn.metrics. See "get_score_names"
score_functions = ['accuracy', 'neg_log_loss', 'roc_auc', 'f1_micro', 'f1_macro']
val_size = .2
train_size =.6

#Classifier type:
#Options: "single-target"
classifier_type = 'single_target'
activation = "sigmoid"

#Random Seed
#controls: train/test/split
#**shuffle in batch splits should stay randommized**
random_seed = 42

#Experiment Files
experiment_name = "test"
experiment_dir = os.path.join(experiments_dir, experiment_name)

#Enable Ray-Tune:
ray_tune = True

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token

# These are the global names you want to skip when saving.
_CONFIG_BLACKLIST = {
   #  "repo_dir",
   #  "corpus_dir",
   #  "train_path",
   #  "word2index_path",
   #  "index2word_path",
   #  "experiments_dir",
   #  "fine_raw_dir",
   #  "fine_processed_dir",
   #  "experiment_dir",
   #  "PAD_token", "SOS_token", "EOS_token", "UNK_token",
   #  'data_dir',
   #  'save_dir_pretrain',
   #  'save_dir_finetune',
   #  'corpora'
}

if __name__ == "__main__":
   print(repo_dir)
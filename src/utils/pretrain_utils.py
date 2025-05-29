from model.data import createTrainFile
from runners.build_vocabulary_objects import build_Vocab
from model.config import *

createTrainFile(pretrain_exclude_phrases)
build_Vocab()


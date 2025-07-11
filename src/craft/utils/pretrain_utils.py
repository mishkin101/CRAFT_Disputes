from craft.model.data import createTrainFile
from craft.runners.build_vocabulary_objects import build_Vocab
from craft.model.config import *

createTrainFile(pretrain_exclude_phrases)
build_Vocab()


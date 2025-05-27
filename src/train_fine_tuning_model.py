
# import all configuration variables
from model.config import *
# import data preprocessing functions
from model.data import *
# import our custom PyTorch modules
from model.model import *

""" 
1.Data Pre-processing: 

    processDialog()
        -  tokenization of texts in eatch batch

    loadPairs()
        -   loading pairts according to the split they are in as (context, reply, label, comment_id). 
        -   Called by processDialog()

2.Loading Corpus Objects: 

    load all corpora into memory

3.Load Vocabulary Object:
    loadPrecomputedVoc()

4.Build Classifiers:
    NEW Predictor(nn.Module): - Subjecitve outcomes *** TO MAKE ***
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
    evaluateBatch()
    - returns the (predicitons) and scores for test/val tests
    
8. Build validate

    
"""

if __name__ == "__main__":
    


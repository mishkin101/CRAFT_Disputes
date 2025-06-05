
import torch
import torch.nn as nn
import torch.nn.functional as F



class AttnSingleTargetClf(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttnSingleTargetClf, self).__init__()
        
        self.hidden_size = hidden_size
        
        # initialize attention
        self.attn = nn.Linear(hidden_size, 1)
        
        # initialize classifier
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, encoder_outputs):
        # compute attention weights
        self.attn_weights = self.attn(encoder_outputs).squeeze().transpose(0,1)
        # softmax normalize weights
        self.attn_weights = F.softmax(self.attn_weights, dim=1).unsqueeze(1)
        # transpose context encoder outputs so we can apply batch matrix multiply
        encoder_outputs_transp = encoder_outputs.transpose(0,1)
        # compute weighted context vector
        context_vec = torch.bmm(self.attn_weights, encoder_outputs_transp).squeeze()
        # forward pass through hidden layers
        layer1_out = self.layer1_act(self.layer1(self.dropout(context_vec)))
        layer2_out = self.layer2_act(self.layer2(self.dropout(layer1_out)))
        # compute and return logits
        logits = self.clf(self.dropout(layer2_out)).squeeze()
        return logits

class SingleTargetClf(nn.Module):
    """
    Single-target classifier head with no attention layer (predicts only from
    the last state vector of the RNN)
    """
    def __init__(self, hidden_size, dropout=0.1):
        super(SingleTargetClf, self).__init__()
        
        self.hidden_size = hidden_size
        
        # initialize classifier
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, encoder_outputs, encoder_input_lengths):
        # from stackoverflow (https://stackoverflow.com/questions/50856936/taking-the-last-state-from-bilstm-bigru-in-pytorch)
        # First we unsqueeze seqlengths two times so it has the same number of
        # of dimensions as output_forward
        # (batch_size) -> (1, batch_size, 1)
        lengths = encoder_input_lengths.unsqueeze(0).unsqueeze(2)
        # Then we expand it accordingly
        # (1, batch_size, 1) -> (1, batch_size, hidden_size) 
        lengths = lengths.expand((1, -1, encoder_outputs.size(2)))

        # take only the last state of the encoder for each batch
        last_outputs = torch.gather(encoder_outputs, 0, lengths-1).squeeze()
        # forward pass through hidden layers
        layer1_out = self.layer1_act(self.layer1(self.dropout(last_outputs)))
        layer2_out = self.layer2_act(self.layer2(self.dropout(layer1_out)))
        # compute and return logits
        logits = self.clf(self.dropout(layer2_out)).squeeze()
        return logits
    

"""================sklearn classifier======================="""
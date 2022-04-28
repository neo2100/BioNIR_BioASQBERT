import torch

# https://github.com/dimartinot/Text-Semantic-Similarity/blob/master/src/model/contrastiveModel.py
class LstmNet(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16):
        super(LstmNet, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim1, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(2*hidden_dim1, hidden_dim2, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm3 = torch.nn.LSTM(2*hidden_dim2 , hidden_dim3, dropout=0.2, batch_first=True, bidirectional=True)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        
    def forward(self, sequence):
        sequence.to(self.device)
        x, _ = self.lstm(sequence)
        
        x = torch.nn.ReLU()(x)
        
        x, _ = self.lstm2(x)
        
        x = torch.nn.ReLU()(x)
        
        _, (h, _) = self.lstm3(x)
        
        return h[-1,:,:]
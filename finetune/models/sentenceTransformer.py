import torch
from torch.nn import functional as F

class SentenceTransformer(torch.nn.Module):

    def __init__(self, baseModel, tokenizer):
        super(SentenceTransformer, self).__init__()
        self.net = baseModel
        self.tokenizer = tokenizer

    def forward(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        model_output = self.net(**encoded_input, return_dict=True)
        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    
    #Mean Pooling - Take average of all tokens
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# input: Model(or checkpoint) + dataset + directory
# output: save the finetuned model

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from tqdm.auto import tqdm

class TripleFinetune:
    def __init__(self, modelCheckPoint, directory):
        # Prepare the model
        self.checkpoint = modelCheckPoint
        self.model = AutoModel.from_pretrained(self.checkpoint, return_dict=True)
        # Put the model in train mode
        self.model.train()
        # To use GPU if it is available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        # To use AdamW iptimizer and set learning rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        # To tokenize and encode the sentences
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        # loss function
        self.calculateLoss = self.triple_loss_function
        self.epsilon = 1

        # directory
        self.directory = directory

    def trainLoop(self, dataset, Testing = False):
        self.dataset = dataset
        progress_bar = tqdm(range(self.dataset.__len__()))
        for textTriple in self.dataset:
            encodedTriple = self.encode([textTriple['anchor'], textTriple['positive'], textTriple['negative']])
            # To calculate loss and update model manually
            loss = self.calculateLoss(encodedTriple[0], encodedTriple[1], encodedTriple[2], self.epsilon)
            if Testing:
                print(loss, loss.shape)
            self.optimizer.zero_grad() # ????
            loss.backward()
            self.optimizer.step()
            # to update progress bar one step forward
            progress_bar.update(1)
        # To save the model
        torch.save({
        	'tokenizer': self.tokenizer,
        	'model_state_dict': self.model.state_dict()},
        	self.directory)


    def  triple_loss_function(self, s_a, s_p, s_n, epsilon):
        # ||sa − sp|| − ||sa − sn|| + epsilon
        distanceDifference = (s_a - s_p).pow(2).sum(-1).sqrt() - (s_a - s_n).pow(2).sum(-1).sqrt() + epsilon
        loss = torch.max(distanceDifference, torch.tensor(0))
        return loss

    #Encode text
    def encode(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        model_output = self.model(**encoded_input, return_dict=True)
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
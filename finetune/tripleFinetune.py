# input: Model(or checkpoint) + dataset + directory
# output: save the finetuned model

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from .models.losses import TripletLoss
from .models.sentenceTransformer import SentenceTransformer
from .models.siameseNetworks import TripletSiamese

class TripleFinetune:
    def __init__(self, modelCheckPoint, inputFile, directory):
        # Prepare the model
        self.checkpoint = modelCheckPoint
        baseModel = AutoModel.from_pretrained(self.checkpoint, return_dict=True)
        # Load model if inputfile
        if inputFile:
            checkpoint = torch.load(inputFile, map_location='cpu')
            tokenizer = checkpoint['tokenizer']
            baseModel.load_state_dict(checkpoint['model_state_dict'])
        else:
            # To tokenize and encode the sentences
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        # make simese network
        transformerModel = SentenceTransformer(baseModel, tokenizer)
        self.model = TripletSiamese(transformerModel)
        
        # Put the model in train mode
        self.model.train()
        # To use GPU if it is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        # To use AdamW iptimizer and set learning rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        # loss function
        self.calculateLoss = TripletLoss(epsilon = 1.0)

        # directory
        self.inputFile = inputFile
        self.directory = directory

    def trainLoop(self, dataset, startIndex, saveInterval, Testing = False):
        self.dataset = dataset
        progress_bar = tqdm(range(self.dataset.__len__()))
        progress_bar.update(startIndex)
        n = 0
        total_loss = 0
        for index in range(startIndex, self.dataset.__len__()):
            textTriple = self.dataset[index]
            self.model.zero_grad()
            # Encoding sentences
            encodedTriple = {}
            encodedTriple['anchor'], encodedTriple['positive'], encodedTriple['negative'] = \
                self.model(textTriple['anchor'], textTriple['positive'], textTriple['negative'])
            # To calculate loss and update model manually
            loss = self.calculateLoss(encodedTriple['anchor'], encodedTriple['positive'], encodedTriple['negative'])
            if Testing:
                print(loss, loss.shape)

            total_loss += loss.item()
            n+=1

            loss.backward()
            self.optimizer.step()
            #self.optimizer.zero_grad() # ????
            # to update progress bar one step forward
            progress_bar.update(1)
            #save in save interval
            if ((index+1) % saveInterval)==0:
                # To save the model
                #torch.save({
                #	'tokenizer': self.model.net.tokenizer,
                #	'model_state_dict': self.model.net.net.state_dict()},
                #	self.directory+str(index))
                print("Last saved index: ", index)
                self.evaluation(index+1-saveInterval, index+1)
                
                print(F"\r Training: Epochs {0} - Val_loss: {total_loss/n} - Batch: {n}/{5}")
                n = 0
                total_loss = 0

        # To save the model
        torch.save({
        	'tokenizer': self.model.net.tokenizer,
        	'model_state_dict': self.model.net.net.state_dict()},
        	self.directory)

    def evaluation(self, startIndex, endIndex):
        n = 1
        total_loss = 0
        num_val_batch = endIndex - startIndex
        with torch.no_grad():
            self.model.eval()
            for index in range(startIndex, endIndex):
                textTriple = self.dataset[index]
                # Encoding sentences
                encodedTriple = {}
                encodedTriple['anchor'], encodedTriple['positive'], encodedTriple['negative'] = \
                    self.model(textTriple['anchor'], textTriple['positive'], textTriple['negative'])
                # To calculate loss and update model manually
                loss = self.calculateLoss(encodedTriple['anchor'], encodedTriple['positive'], encodedTriple['negative'])
                total_loss += loss.item()

                acc=0
                print(F"\r Epochs {0} - Val_loss: {total_loss/n} - Batch: {n}/{num_val_batch}")
                n+=1
            
            #scheduler.step(total_loss)
            self.model.train()
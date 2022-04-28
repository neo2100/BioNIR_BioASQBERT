# input: Model(or checkpoint) + dataset + directory
# output: save the finetuned model

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from .models.losses import TripletLoss, QuadrupletLoss
from .models.sentenceTransformer import SentenceTransformer
from .models.siameseNetworks import TripletSiamese, QuadrupletSiamese
from .models.bionirHeads import LstmNet

class FinetuneHeads:
    def __init__(self, networkModel, epsilon, learningRate, modelCheckPoint, inputFile, directory):
        # Prepare the model
        self.checkpoint = modelCheckPoint
        baseModel = AutoModel.from_pretrained(self.checkpoint, return_dict=True)
        # To tokenize and encode the sentences
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        # Load model if inputfile
        #if inputFile:
        #    checkpoint = torch.load(inputFile, map_location='cpu')
        #    tokenizer = checkpoint['tokenizer']
        #    baseModel.load_state_dict(checkpoint['model_state_dict'])

        # make simese network
        self.transformerModel = SentenceTransformer(baseModel, tokenizer, noGrad = True)
        bionirHead = LstmNet(768, 48, 32, 16)
        if networkModel == "Triple":
            self.model = TripletSiamese(bionirHead)
            self.calculateLoss = TripletLoss(epsilon['epsilon1']) # loss function
        else:
            self.model = QuadrupletSiamese(bionirHead) 
            self.calculateLoss = QuadrupletLoss(epsilon['epsilon1'], epsilon['epsilon2']) # loss function
        
        # Put the model in train mode
        self.model.train()
        # To use GPU if it is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Device', self.device)
        self.model.to(self.device)
        self.transformerModel.to(self.device) 
        # To use AdamW iptimizer and set learning rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learningRate)

        # directory
        self.inputFile = inputFile
        self.directory = directory
        self.trainingLosses = []
        self.evaluationLosses = []

    def trainLoop(self, dataset, startIndex, endIndex, evaluateInterval, saveInterval, Testing = False):
        self.dataset = dataset
        #   reseting
        self.trainingLosses = []
        self.evaluationLosses = []

        progress_bar = tqdm(range(self.dataset.__len__()))
        progress_bar.update(startIndex)
        total_loss = 0
        for index in range(startIndex, min(endIndex,self.dataset.__len__())):
            textBundle = self.dataset[index]
            self.model.zero_grad()
            # Transform Sentences
            embedBundle = {}
            embedBundle['anchor'] = self.transformerModel(textBundle['anchor']).view(-1, 1, 768)
            embedBundle['positive'] = self.transformerModel(textBundle['positive']).view(-1, 1, 768)
            embedBundle['negative1'] = self.transformerModel(textBundle['negative1']).view(-1, 1, 768)
            # Encoding sentences
            encoded =  self.model(embedBundle)
            # To calculate loss and update model manually
            loss = self.calculateLoss(encoded)
            if Testing:
                print(loss, loss.shape)

            #if loss.item()<0.1:
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            #self.optimizer.zero_grad() # ?
            # to update progress bar one step forward
            progress_bar.update(1)
            #save in save interval
            if ((index+1) % saveInterval)==0:
                # To save the model
                torch.save( self.model.net.state_dict(),
                	self.directory+'_Interval_'+str(index+1))
                print("Last saved index: ", index)
            # evaluate in evaluate interval
            if ((index+1) % evaluateInterval)==0:
                print(F"\r Training: Epochs {(index+1)/evaluateInterval} - Val_loss: {total_loss/evaluateInterval} - index: {index+1} ")
                self.trainingLosses.append(str(total_loss/evaluateInterval))
                total_loss = 0

                self.evaluation(index+1-evaluateInterval, index+1)

        # To save the model
        torch.save(self.model.net.state_dict(),
        	self.directory+'_Ended_'+str(endIndex))
        # print losses formatted
        print("Training Loss:", ','.join(self.trainingLosses))
        print("Evaluation Loss:", ','.join(self.evaluationLosses))

    def evaluation(self, startIndex, endIndex):
        n = 0
        total_loss = 0
        num_val_batch = endIndex - startIndex
        with torch.no_grad():
            self.model.eval()
            for index in range(startIndex, endIndex):
                textBundle = self.dataset[index]
                # Transform Sentences
                embedBundle = {}
                embedBundle['anchor'] = self.transformerModel(textBundle['anchor']).view(-1, 1, 768)
                embedBundle['positive'] = self.transformerModel(textBundle['positive']).view(-1, 1, 768)
                embedBundle['negative1'] = self.transformerModel(textBundle['negative1']).view(-1, 1, 768)
                # Encoding sentences
                encoded = self.model(embedBundle)
                # To calculate loss and update model manually
                loss = self.calculateLoss(encoded)
                total_loss += loss.item()
                n+=1
            
            print(F"\r Evaluation: Epochs {endIndex/num_val_batch} - Val_loss: {total_loss/n} - Batch: {n}/{num_val_batch}")
            self.evaluationLosses.append(str(total_loss/n))

            #scheduler.step(total_loss)
            self.model.train()
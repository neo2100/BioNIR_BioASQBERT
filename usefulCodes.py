
# To get model and put it in train mode
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
model.train()


# To use GPU if it is available
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# To use AdamW iptimizer and set learning rate
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)


# To tokenize and encode the sentences
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']


# To calculate loss and update model manually
from torch.nn import functional as F
labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask)
loss = F.cross_entropy(labels, outputs.logits)
loss.backward()
optimizer.step()


# To use scheduler and warm up
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
#compute the gradients, and update the model parameters,Zero the optimizer, 
loss.backward()
optimizer.step()
scheduler.step()
### Still don't know WHY?
optimizer.zero_grad()

# To save the model
torch.save({
	'tokenizer': tokenizer,
	'model_state_dict': model.state_dict()},
	model+".pt")


# To have progress bar
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

## to update progress bar one step forward
progress_bar.update(1)

# To make loss function
#Name: 		binary_cross_entropy
#Purpose: 	defines binary cross entropy loss function
#Inputs: 	predictions -> model predictions
# 			targets -> target labels
#Outputs: 	loss -> loss value
def  binary_cross_entropy(predictions, targets):
	loss =  -(targets * torch.log(predictions) + (1  - targets) * torch.log(1  - predictions))
	loss = torch.mean(loss)
	return loss
criterion = binary_cross_entropy
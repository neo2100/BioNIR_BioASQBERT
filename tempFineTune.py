import torch
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.nn import functional as F

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings



#######   TRAINING PART
# The loss function
def  triple_loss_function(s_a, s_p, s_n, epsilon):
    # ||sa − sp|| − ||sa − sn|| + epsilon
    distanceDifference = (s_a - s_p).pow(2).sum(-1).sqrt() - (s_a - s_n).pow(2).sum(-1).sqrt() + epsilon
    loss = max(distanceDifference, torch.tensor(0))
    return loss


# To get model and put it in train mode
checkpoint = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
model = AutoModel.from_pretrained(checkpoint, return_dict=True)
model.train()

# To use GPU if it is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# To use AdamW iptimizer and set learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# To tokenize and encode the sentences
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# loss function
criterion = triple_loss_function
epsilon = 1

# a, p, n
text_triple = ["How do you feel about Pixar?", "I don't care for Pixar.", "I like footbal."]

encodedTriple = encode(text_triple)

# To calculate loss and update model manually
loss = criterion(encodedTriple[0],encodedTriple[1],encodedTriple[2],epsilon)
print(loss, loss.shape)
# optimizer.zero_grad() # ????
loss.backward()
optimizer.step()


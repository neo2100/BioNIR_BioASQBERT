import os
import json
from finetune.finetune import Finetune

inputFileName = './dataset/training9b_quadruples.json'

# preparing inputs
dataset = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, inputFileName)
with open(abs_file_path, 'rb') as document_file:
    dataset = json.load(document_file)['quadruples']
    document_file.close()

# finetune = Finetune(
#     networkModel="Triple",
#     epsilon={
#         'epsilon1': 0.5
#     },
#     learningRate=2e-6,
#     modelCheckPoint="sentence-transformers/multi-qa-mpnet-base-cos-v1",
#     inputFile=None,
#     directory='sbert_nn6.4_qu.pt')

finetune = Finetune(
     networkModel="Quadruple",
     epsilon={
         'epsilon1': 0.5,
         'epsilon2': 0.25
     },
     learningRate=1e-5,
     modelCheckPoint="sentence-transformers/multi-qa-mpnet-base-cos-v1",
     inputFile=None,
     directory='sbert_nn7.3_qu.pt')

#finetune.trainLoop(dataset, 6000, 8000, 500, 2000, False)
finetune.trainLoop(dataset, 0, 46130, 500, 46131, False)

# nn6.0: Triple, epsilon 0.5, lr=2e-5    #gpu018
# nn6.1: Triple, epsilon 1.0, lr=2e-5
# nn6.2: Triple, epsilon 1.0, lr=1e-5
# nn6.3: Triple, epsilon 0.5, lr=1e-5
# nn6.4: Triple, epsilon 0.5, lr=2e-6
# nn7.0: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=2e-5
# nn7.1: Quadruple, epsilon1 = 1.0, epsilon2 = 0.5, lr=2e-5
# nn7.3: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=1e-5
import os
import json
from finetune.finetuneHeads import FinetuneHeads

inputFileName = './dataset/training9b_quadruples.json'

# preparing inputs
dataset = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, inputFileName)
with open(abs_file_path, 'rb') as document_file:
    dataset = json.load(document_file)['quadruples']
    document_file.close()

finetune = FinetuneHeads(
     networkModel="Triple",
     epsilon={
         'epsilon1': 1
     },
     learningRate=1e-3,
     modelCheckPoint="sentence-transformers/multi-qa-mpnet-base-cos-v1",
     inputFile=None,
     directory='bionirhead_nn0_qu.pt')

# finetune = FinetuneHeads(
#      networkModel="Quadruple",
#      epsilon={
#          'epsilon1': 0.5,
#          'epsilon2': 0.25
#      },
#      learningRate=1e-5,
#      modelCheckPoint="sentence-transformers/multi-qa-mpnet-base-cos-v1",
#      inputFile=None,
#      directory='sbert_nn7.83_qu.pt')

#finetune.trainLoop(dataset, 0, 8000, 500, 2000, True)
finetune.trainLoop(dataset, 0, 46130, 500, 10, False)

# nn6.0: Triple, epsilon 0.5, lr=2e-5    #gpu018
# nn6.1: Triple, epsilon 1.0, lr=2e-5
# nn6.2: Triple, epsilon 1.0, lr=1e-5
# nn6.3: Triple, epsilon 0.5, lr=1e-5
# nn6.4: Triple, epsilon 0.5, lr=2e-6
# nn6.5: Triple, epsilon 0.5, lr=1e-6
# nn6.6: Triple, epsilon 0.25, lr=1e-5  #5584172
# nn6.7: Triple, epsilon 0.0, lr=1e-5   #5584847 gpu014
# nn6.8: Triple, epsilon 0.5, lr=1e-5   #5584937 gpu020  <0.25 -> = 0
# nn6.81: Triple, epsilon 0.5, lr=1e-5   #5584957 gpu014  <0.1 -> = 0
# nn6.82: Triple, epsilon 0.5, lr=1e-5   #5585048 gpu020  <0.25 -> = 0.25
# nn6.83: Triple, epsilon 0.5, lr=1e-5   #5585052 gpu023  <0.1 -> = 0.1
# nn7.0: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=2e-5
# nn7.1: Quadruple, epsilon1 = 1.0, epsilon2 = 0.5, lr=2e-5
# nn7.3: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=1e-5
# nn7.4: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=2e-6
# nn7.5: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=1e-6
# nn7.6: Quadruple, epsilon1 = 0.25, epsilon2 = 0.25, lr=1e-5  #5584173
# nn7.7: Quadruple, epsilon1 = 0.0, epsilon2 = 0.0, lr=1e-5     #5584853 gpu015
# nn7.82: Quadruple, epsilon1 = 0.5, epsilon2 = 0.25, lr=1e-5     #5585156 gpu015 <0.25 -> = 0.25
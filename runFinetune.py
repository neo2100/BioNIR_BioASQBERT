import os
import json
from finetune.tripleFinetune import TripleFinetune

inputFileName = './dataset/training9b_quadruples.json'

# preparing inputs
dataset = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, inputFileName)
with open(abs_file_path, 'rb') as document_file:
    dataset = json.load(document_file)['quadruples']
    document_file.close()


tripleFinetune = TripleFinetune(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1", 'sbert_nn6_qu_8000.pt', 'sbert_nn6_qu.pt')

#tripleFinetune.trainLoop(dataset, 6000, 8000, 500, 2000, False)
tripleFinetune.trainLoop(dataset, 8000, 46130, 500, 2000, False)
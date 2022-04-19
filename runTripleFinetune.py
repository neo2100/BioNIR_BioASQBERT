import os
import json
from finetune.tripleFinetune import TripleFinetune

inputFileName = './dataset/training9b_triples.json'

# preparing inputs
dataset = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, inputFileName)
with open(abs_file_path, 'rb') as document_file:
    dataset = json.load(document_file)['triples']
    document_file.close()


tripleFinetune = TripleFinetune(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1", None, 'sbert_sample.pt')

tripleFinetune.trainLoop(dataset, 0, 500, False)
import os
import json
from tqdm.auto import tqdm
from random import randrange
inputFileName = 'training9b.json'

def findNegativeSnippet(index, maxQuestionNum, questions):
    randomIndex = randrange(0, maxQuestionNum, 1)
    while randomIndex==index:
        randomIndex = randrange(0, maxQuestionNum, 1)
    if randomIndex==index:
        print("ERROR: Same index happend in index " + str(index))

    return questions[randomIndex]['snippets'][randrange(0, questions[randomIndex]['snippets'].__len__(), 1)]["text"]


# preparing inputs
questions = {}
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, inputFileName)
with open(abs_file_path, 'rb') as document_file:
    questions = json.load(document_file)['questions']
    document_file.close()

triples = []
maxQuestionNum = questions.__len__()
progress_bar = tqdm(range(maxQuestionNum))
for index, question in enumerate(questions):
    for snippet in question["snippets"]:
        triple = {}
        triple['anchor'] = question["body"]
        triple['positive'] = snippet["text"]
        triple['negative'] = findNegativeSnippet(index, maxQuestionNum, questions)
        triples.append(triple)
     
    
    # to update progress bar one step forward
    progress_bar.update(1)

print(triples.__len__())
# saving outputs in a file
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, (inputFileName+'_triples.json'))
# saving outputs in a file    
with open(abs_file_path, 'w', encoding='utf-8') as document_file:
    json.dump({'triples': triples},
              document_file, ensure_ascii=False, indent=4)
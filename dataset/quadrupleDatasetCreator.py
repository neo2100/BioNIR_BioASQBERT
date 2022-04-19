import os
import json
from tqdm.auto import tqdm
from random import randrange, shuffle
inputFileName = 'training9b.json'

def findNegativeSnippets(index, maxQuestionNum, questions):
    randomIndex1 = randrange(0, maxQuestionNum, 1)
    while randomIndex1==index:
        randomIndex1 = randrange(0, maxQuestionNum, 1)
        
    randomIndex2 = randrange(0, maxQuestionNum, 1)
    while randomIndex2==index or randomIndex2==randomIndex1:
        randomIndex2 = randrange(0, maxQuestionNum, 1)

    if randomIndex1==index or randomIndex2==index or randomIndex1==randomIndex2:
        print("ERROR: Same index happend in index " + str(index))

    negative1 = questions[randomIndex1]['snippets'][randrange(0, questions[randomIndex1]['snippets'].__len__(), 1)]["text"]
    negative2 = questions[randomIndex2]['snippets'][randrange(0, questions[randomIndex2]['snippets'].__len__(), 1)]["text"]

    return negative1, negative2


# preparing inputs
questions = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, inputFileName)
with open(abs_file_path, 'rb') as document_file:
    questions = json.load(document_file)['questions']
    document_file.close()

# shuffle the sentences
shuffle(questions)

quadruples = []
maxQuestionNum = questions.__len__()
progress_bar = tqdm(range(maxQuestionNum))
for index, question in enumerate(questions):
    for snippet in question["snippets"]:
        quadruple = {}
        quadruple['anchor'] = question["body"]
        quadruple['positive'] = snippet["text"]
        quadruple['negative1'], quadruple['negative2'] = findNegativeSnippets(index, maxQuestionNum, questions)
        quadruples.append(quadruple)
     
    
    # to update progress bar one step forward
    progress_bar.update(1)

# Shuffle them all
shuffle(quadruples)

print(quadruples.__len__())
# saving outputs in a file
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
abs_file_path = os.path.join(script_dir, (inputFileName+'_quadruples.json'))
# saving outputs in a file    
with open(abs_file_path, 'w', encoding='utf-8') as document_file:
    json.dump({'quadruples': quadruples},
              document_file, ensure_ascii=False, indent=4)
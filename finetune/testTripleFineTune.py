from tripleFinetune import TripleFinetune

tripleFinetune = TripleFinetune(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1", '.\\finetune\models\sbert.pt')

dataset = [{
    'anchor': "How do you feel about Pixar?",
    'positive': "I don't care for Pixar.",
    'negative': "I like footbal."
},{
    'anchor': "How do you feel about Pixar?",
    'positive': "I don't care for Pixar.",
    'negative': "I like footbal."
}]

tripleFinetune.trainLoop(dataset, True)
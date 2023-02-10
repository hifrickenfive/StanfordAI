from nltk.translate.bleu_score import sentence_bleu, ngrams
reference = [
    'resources have to be sufficient and they have to be predictable'.split(),
    'adequate and predictable resources are required'.split(),
]

c1 = 'there is a need for adequate and predictable resources'.split()
c2 = 'resources be sufficient and predictable to'.split()


print('BLEU score -> {}'.format(sentence_bleu(reference, c1, weights=(0.5, 0.5, 0, 0))))



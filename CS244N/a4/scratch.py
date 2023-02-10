def count_bigrams(sentence):
    words = sentence.split()
    bigrams = {}
    for w1, w2 in zip(words, words[1:]):
        bigram = f"{w1} {w2}"
        if bigram in bigrams:
            bigrams[bigram] += 1
        else:
            bigrams[bigram] = 1
    return bigrams

def count_common_bigrams(sentence1, sentence2):
    bigrams1 = count_bigrams(sentence1)
    bigrams2 = count_bigrams(sentence2)
    bigram_count = 0
    common_bigram = list()
    for bigram, count in bigrams1.items():
        if bigram in bigrams2:
            bigram_count += count
            common_bigram.append(bigram)
    return bigram_count, common_bigram

r1 = "resources have to be sufficient and they have to be predictable"
r2 = 'adequate and predictable resources are required'
c1 = "there is a need for adequate and predictable resources"
c2 = 'resources be sufficient and predictable to'

r1_bigrams = count_bigrams(r1)
c1_bigrams = count_bigrams(c1)

bigram_count, common_bigram = count_common_bigrams(c1, r1)
print(bigram_count, common_bigram)
bigram_count, common_bigram = count_common_bigrams(c1, r2)
print(bigram_count, common_bigram)

bigram_count, common_bigram = count_common_bigrams(c2, r1)
print(bigram_count, common_bigram)
bigram_count, common_bigram = count_common_bigrams(c2, r2)
print(bigram_count, common_bigram)

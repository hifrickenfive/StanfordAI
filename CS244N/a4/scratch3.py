import numpy as np
from nltk import ngrams

r1 = 'resources have to be sufficient and they have to be predictable'
r2 = 'adequate and predictable resources are required'
c1 = 'there is a need for adequate and predictable resources'
c2 = 'resources be sufficient and predictable to'

def two_grams(s):
    words = s.split()
    return [f"{words[i]} {words[i + 1]}" for i in range(0, len(words) - 1, 1)]


r1_twograms = ['resources have', 'have to', 'to be', 'be sufficient', 'sufficient and', 'and they', 'they have', 'have to', 'to be', 'be predictable']
r2_twograms = ['adequate and', 'and predictable', 'predictable resources', 'resources are', 'are required']
c1_twograms = ['there is', 'is a', 'a need', 'need for', 'for adequate', 'adequate and', 'and predictable', 'predictable resources']
c2_twograms = ['resources be', 'be sufficient', 'sufficient and', 'and predictable', 'predictable to']

r1_twograms2 = two_grams(r1)
r2_twograms2 =two_grams(r2)
c1_twograms2 =two_grams(c1)
c2_twograms2 = two_grams(c2)


def count2grams(candidate_2grams, ref_2grams):
    twograms_count_in_ref = 0
    for grams in candidate_2grams:
        if grams in ref_2grams:
            twograms_count_in_ref += 1
    return twograms_count_in_ref

c1_twograms_in_r1 = count2grams(c1_twograms, r1_twograms)
c1_twograms_in_r2 = count2grams(c1_twograms, r2_twograms)
c2_twograms_in_r1 = count2grams(c2_twograms, r1_twograms)
c2_twograms_in_r2 = count2grams(c2_twograms, r2_twograms)

c1_twograms_in_r12 = count2grams(c1_twograms2, r1_twograms2)
c1_twograms_in_r22 = count2grams(c1_twograms2, r2_twograms2)
c2_twograms_in_r12 = count2grams(c2_twograms2, r1_twograms2)
c2_twograms_in_r22 = count2grams(c2_twograms2, r2_twograms2)


r1 = 'resources have to be sufficient and they have to be predictable'.split()
r2 = 'adequate and predictable resources are required'.split()
c1 = 'there is a need for adequate and predictable resources'.split()
c2 = 'resources be sufficient and predictable to'.split()

def countunigrams(candidate, ref):
    c_unigram_count_in_ref = 0
    for grams in candidate:
        if grams in ref:
            c_unigram_count_in_ref += 1
    return c_unigram_count_in_ref

c1_unigrams_in_r1 = countunigrams(c1, r1)
c1_unigrams_in_r2 = countunigrams(c1, r2)
c2_unigrams_in_r1 = countunigrams(c2, r1)
c2_unigrams_in_r2 = countunigrams(c2, r2)

def calc_BLEU(BP, p1, p2, lam):
    return BP*np.exp(lam*np.log(p1) + lam*np.log(p2))

def solve(count_unigram, count_ngram, candidate_sentence, len_twogram, closest_ref_sentence, lam):
    p1 = count_unigram/len(candidate_sentence)
    p2 = count_ngram/len_twogram
    print(f'p1= {p1} and p2 ={p2}')
    if len(candidate_sentence)>= len(closest_ref_sentence):
        BP = 1
    else:
        BP = np.exp(1- len(r1)/len(candidate_sentence))
    print(f'BP = {BP}')
    BLEU = calc_BLEU(BP, p1, p2, lam)
    print(f'BLEU = {BLEU}')
    print('\n')

lam = 0.5

print(f'c1 unigram count in r1: {c1_unigrams_in_r1}')
print(f'c1 2gram count in r1: {c1_twograms_in_r1}')
print(f'c1 unigram count in r2: {c1_unigrams_in_r2}')
print(f'c1 2gram count in r2: {c1_twograms_in_r2}')

print(f'c2 unigram count in r1: {c2_unigrams_in_r1}')
print(f'c2 2gram count in r1: {c2_twograms_in_r1}')
print(f'c2 unigram count in r2: {c2_unigrams_in_r2}')
print(f'c2 2gram count in r2: {c2_twograms_in_r2}')

result_c1 = solve(
    max(c1_unigrams_in_r1, c1_unigrams_in_r2), 
    max(c1_twograms_in_r1, c1_twograms_in_r2),
    c1, 
    len(c1_twograms), 
    r1, lam
)

result_c2 = solve(
    max(c2_unigrams_in_r1, c2_unigrams_in_r2), 
    max(c2_twograms_in_r1, c2_twograms_in_r2),
    c2, 
    len(c2_twograms), 
    r2, lam
)

result_c1 = solve(
    c1_unigrams_in_r2, 
    c1_twograms_in_r2,
    c1, 
    len(c1_twograms), 
    r2, lam
)

result_c2 = solve(
    c2_unigrams_in_r2, 
    c2_twograms_in_r2,
    c2, 
    len(c2_twograms), 
    r2, lam
)
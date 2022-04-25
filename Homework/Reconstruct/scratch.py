import util
import submission
import wordsegUtil
import graderUtil
import sys

CORPUS = 'leo-will.txt'

QUERIES_BOTH = [
    'stff',
    'hllthr',
    'thffcrndprncndrw',
    'ThstffffcrndPrncndrwmntdthrhrssndrdn',
    'whatsup',
    'ipovercarrierpigeon',
    'aeronauticalengineering',
    'themanwiththegoldeneyeball',
    'lightbulbsneedchange',
    'internationalplease',
    'comevisitnaples',
    'somethingintheway',
    'itselementarymydearwatson',
    'itselementarymyqueen',
    'themanandthewoman',
    'nghlrdy',
    'jointmodelingworks',
    'jointmodelingworkssometimes',
    'jointmodelingsometimesworks',
    'rtfclntllgnc',
]




def getRealCosts():

    global _realUnigramCost, _realBigramCost, _possibleFills
    _realUnigramCost, _realBigramCost, _possibleFills = None, None, None

    if _realUnigramCost is None:
        sys.stdout.write('Training language cost functions [corpus: %s]... ' % CORPUS)
        sys.stdout.flush()

        _realUnigramCost, _realBigramCost = wordsegUtil.makeLanguageModels(CORPUS)
        _possibleFills = wordsegUtil.makeInverseRemovalDictionary(CORPUS, 'aeiou')

        print('Done!')
        print('')

    return _realUnigramCost, _realBigramCost, _possibleFills

def t_3b_5():

    unigramCost, bigramCost, possibleFills = getRealCosts()

    smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
    for query in QUERIES_BOTH:
        query = wordsegUtil.cleanLine(query)
        parts = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]
        pred = [submission.segmentAndInsert(part, smoothCost, possibleFills) for part in parts]

        print(query)
        print(parts)
        print(pred)
        print('\n')


def t_3b_3():
    unigramCost, bigramCost, possibleFills = getRealCosts()

    bigramCost = lambda a, b: unigramCost(b)
    fills_ = {
        'nd': set(['and']),
        'tw': set(['two']),
        'thr': set(['three']),
        'wrd': set(['word']),
        'wrds': set(['words']),
        # Hah!  Hit them with two better words
        'th': set(['the']),
        'rwrds': set(['rewards']),
    }
    fills = lambda x: fills_.get(x, set())

    solution1 = submission.segmentAndInsert('wrd', bigramCost, fills)
    solution2 = submission.segmentAndInsert('twwrds', bigramCost, fills)
    # Waddaya know
    solution3 = submission.segmentAndInsert('ndthrwrds', bigramCost, fills)


    return solution1, solution2, solution3

def t_3b_4():
    def bigramCost(a, b):
        corpus = [wordsegUtil.SENTENCE_BEGIN] + 'beam me up scotty'.split()
        if (a, b) in list(zip(corpus, corpus[1:])):
            return 1.0
        else:
            return 1000.0

    def possibleFills(x):
        fills = {
            'bm'   : set(['beam', 'bam', 'boom']),
            'm'    : set(['me', 'ma']),
            'p'    : set(['up', 'oop', 'pa', 'epe']),
            'sctty': set(['scotty']),
            'z'    : set(['ze']),
        }
        return fills.get(x, set())

    # Ensure no non-word makes it through
    solution1 = submission.segmentAndInsert('zzzzz', bigramCost, possibleFills)
    solution2 = submission.segmentAndInsert('bm', bigramCost, possibleFills)
    solution3 = submission.segmentAndInsert('mp', bigramCost, possibleFills)
    solution4 = submission.segmentAndInsert('bmmpsctty', bigramCost, possibleFills)

    return solution1, solution2, solution3, solution4

if __name__ == '__main__':
    solution1, solution2, solution3, solution4 = t_3b_4()
    print(solution1)
    print(solution2)
    print(solution3)
    print(solution4)


    solution1, solution2, solution3 = t_3b_3()
    print(solution1)
    print(solution2)
    print(solution3)
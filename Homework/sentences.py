import collections

def mutate_sentences(sentence:str):
    word_list = sentence.split()
    MAX_LENGTH = len(word_list)

    word_dict = collections.defaultdict(list) # create lookup
    for i in range(len(word_list)-1):
        word_dict[word_list[i]].append(word_list[i+1])

    def dfs(sentences:list, current_sentence:list, lookup:dict, key:str, current_length:int, MAX_LENGTH:int):
        current_sentence.append(key)
        current_length += 1

        if len(current_sentence) == MAX_LENGTH: # Base case
            sentences.append(' '.join(current_sentence))
            return
        elif key not in lookup.keys(): # Block use of the last word of input sentence in the middle of new sentence
            return
        else:
            for neighbour in lookup[key]:
                if len(lookup[key]) > 1: # Spawn copy of list for each branch
                    dfs(sentences,  current_sentence.copy(), lookup, neighbour, current_length, MAX_LENGTH)
                else:
                    dfs(sentences, current_sentence, lookup, neighbour, current_length, MAX_LENGTH)

    all_sentences = list()
    for key in word_dict.keys():
        subset_of_all_sentences = list()
        current_sentence = list()
        dfs(subset_of_all_sentences, current_sentence, word_dict, key, 0, MAX_LENGTH)
        all_sentences.append(subset_of_all_sentences)
    all_sentences = sum(all_sentences, [])  # Flatten nested list
    return all_sentences

if __name__ == '__main__':
    sentence  = 'the cat and the mouse'
    result = mutate_sentences(sentence)
    print(result)

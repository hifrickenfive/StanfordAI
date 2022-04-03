import collections

def dfs(all_sentences:list, current_sentence:list, graph:dict, key:str, current_length:int, MAX_LENGTH:int):
    current_sentence.append(key)
    current_length += 1

    # Base case: new_sentence length equals original sentence
    if len(current_sentence) == MAX_LENGTH:
        all_sentences.append(' '.join(current_sentence))
        return
    else:
        for neighbour in graph[key]:
            if len(graph[key]) > 1: # Spawn another list for each branch
                dfs(all_sentences,  current_sentence.copy(), graph, neighbour, current_length, MAX_LENGTH)
            else:
                dfs(all_sentences, current_sentence, graph, neighbour, current_length, MAX_LENGTH)


def mutate_sentences(sentence):
    word_list = sentence.split()
    word_dict = collections.defaultdict(list)
    for i in range(len(word_list)-1):
        word_dict[word_list[i]].append(word_list[i+1])

    MAX_LENGTH = len(word_list)

    all_sentences = list()
    for key in word_dict.keys():
        subset_of_all_sentences = list()
        current_sentence = list()
        dfs(subset_of_all_sentences, current_sentence, word_dict, key, 0, MAX_LENGTH)
        all_sentences.append(subset_of_all_sentences)
    
    return all_sentences


if __name__ == '__main__':
    sentence  = 'the cat and the mouse'
    result = mutate_sentences(sentence)
    print(result)

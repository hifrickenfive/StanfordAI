import collections

sentence  = 'the cat and the mouse'
word_list = sentence.split()
word_dict = collections.defaultdict(list)
for i in range(len(word_list)-1):
    word_dict[word_list[i]].append(word_list[i+1])

print(word_dict)

def dfs(visited, graph, node, length):
    # print(length)
    if len(visited) == length:
        return

    # print (node)
    visited.append(node)
    length += 1
    print(visited)
    for neighbour in graph[node]:
        dfs(visited, graph, neighbour, length)


visited = list() # Set to keep track of visited nodes of graph.
dfs(visited, word_dict, 'the', 5)


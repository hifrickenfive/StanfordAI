from attr import has


actions = {'text', 'aeiou'}
actionsFinal = actions.copy()
results = []

for action in actions:
    hasConsonants = False
    for char in action:
        if char not in 'aeiou':
            hasConsonants = True
    print(hasConsonants)
    if not hasConsonants:
        continue
    else:
        results.append(action)
        print(results)


    hasConsonants = False
    for char in action:
        if char not in 'aeiou':
            hasConsonants = True
    if not hasConsonants:
        continue
    else:

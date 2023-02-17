# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

pathToDevSet = "c:\\data\\StanfordAI\\CS244N\\a5\\src\\birth_dev.tsv" # OK to hardcode https://edstem.org/us/courses/33056/discussion/2556803
numDataInDevSet = len(open(pathToDevSet, encoding='utf-8').readlines())
lodonOnlyPredictions = numDataInDevSet * ['London']
total, numMatch = utils.evaluate_places(pathToDevSet, lodonOnlyPredictions)
print(f'{numMatch/total} correct in dev set if all predictions == London') # 0.05

# What about the actual Vanilla Predictions on dev?
# pathToActualDevPredictions = "c:\\data\\StanfordAI\\CS244N\\a5\\vanilla.nopretrain.dev.predictions"
# actualDevPredictions = open(pathToActualDevPredictions, encoding='utf-8').readlines()
# actualDevPredictions = [result.strip() for result in actualDevPredictions]
# total, correct = utils.evaluate_places(pathToDevSet, actualDevPredictions)
# print(f'{correct/total} correct on dev set from actual predictions') # 0.008 :(
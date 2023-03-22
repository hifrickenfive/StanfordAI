# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

# OK to hardcode https://edstem.org/us/courses/33056/discussion/2556803
pathToDevSet = "c:\\data\\StanfordAI\\CS244N\\a5\\src\\birth_dev.tsv"
pathToNoPretrainDevPredict = "c:\\data\\StanfordAI\\CS244N\\a5\\vanilla.nopretrain.dev.predictions" 

numDataInDevSet = len(open(pathToDevSet, encoding='utf-8').readlines())
lodonOnlyPredictions = numDataInDevSet * ['London']
total, numMatch = utils.evaluate_places(pathToDevSet, lodonOnlyPredictions)
print(f'{numMatch/total} correct in dev set if all predictions == London') # 0.05 :/
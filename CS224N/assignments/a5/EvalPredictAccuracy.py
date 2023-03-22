import utils

# OK to hardcode https://edstem.org/us/courses/33056/discussion/2556803
pathToDevSet = "c:\\data\\StanfordAI\\CS244N\\a5\\src\\birth_dev.tsv"
pathToTestSet = "c:\\data\\StanfordAI\\CS244N\\a5\\src\\birth_test_inputs.tsv"
pathToNoPretrainDevPredict = "c:\\data\\StanfordAI\\CS244N\\a5\\vanilla.nopretrain.dev.predictions" 
pathToPretrainDevPredict = "c:\\data\\StanfordAI\\CS244N\\a5\\vanilla.pretrain.dev.predictions" 

def EvalPredictAccuracy(pathToDataSet, pathToPredictions, datasetKind='<datasetKind>', modelKind='<modelKind>'):
    predictions = open(pathToPredictions, encoding='utf-8').readlines()
    predictions = [result.strip() for result in predictions]
    total, correct = utils.evaluate_places(pathToDataSet, predictions)
    print(f'{correct/total} correct on {datasetKind} vs. {modelKind} predictions')

# London only predictions
numDataInDevSet = len(open(pathToDevSet, encoding='utf-8').readlines())
lodonOnlyPredictions = numDataInDevSet * ['London']
total, numMatch = utils.evaluate_places(pathToDevSet, lodonOnlyPredictions)
print(f'{numMatch/total} correct in dev set if all predictions == London') # 0.05 :/

# Predictions from the models
EvalPredictAccuracy(pathToDevSet, pathToPretrainDevPredict, datasetKind='dev', modelKind='pretrained+finetuned') # 0.344 :)
EvalPredictAccuracy(pathToDevSet, pathToNoPretrainDevPredict, datasetKind='dev', modelKind='no-pretrain') # 0.008 :(

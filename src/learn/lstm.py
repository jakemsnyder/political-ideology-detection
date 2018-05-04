from time import time

from keras import Sequential
import numpy as np
import pandas as pd
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing import sequence
from matplotlib import pyplot

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold

MODEL_PATH = '../../data/binary/word2Vec.bin'
# TRAINING_DATA_PATH = '../../data/csv/model/train.csv'
# TEST_DATA_PATH = '../../data/csv/model/test.csv'
# TRAINING_DATA_PATH = '../../data/csv/model/trainFAKEDATA.csv'
TRAINING_DATA_PATH = '../../data/csv/model/trainPolitical.csv'
TEST_DATA_PATH = '../../data/csv/model/testPolitical.csv'
# TRAINING_DATA_PATH = '../../data/csv/model/trainAbortion.csv'
# TEST_DATA_PATH = '../../data/csv/model/testAbortion.csv'
# TRAINING_DATA_PATH = '../../data/csv/model/trainModern.csv'
# TEST_DATA_PATH = '../../data/csv/model/testModern.csv'
DEBATE_DATA_PATH = '../../data/csv/processed/debate_sentences_part1.csv'
RESULT_PATH = '../../results/csv/lstmParameters(topic modelling).csv'
RESULT_FINAL_PATH = '../../results/csv/lstmResults.csv'
# RESULT_DEBATE_FINAL_PATH = '../../results/csv/lstmDebate.csv'
RESULT_DEBATE_FINAL_PATH = '../../results/csv/lstmDebate2.csv'
DIFFERENCES_PATH = '../../results/csv/worstMistakes.csv'


def saveWord2VecModel(savePath, model):
    model.save(savePath)


def loadWord2VecModel(loadPath):
    return Word2Vec.load(loadPath)


def visualizeWordEmbeddings(wordEmbeddingMatrix, model):
    # Taken from https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
    pca = PCA(n_components=2)
    fittedPCA = pca.fit_transform(wordEmbeddingMatrix)
    pyplot.scatter(fittedPCA[:, 0], fittedPCA[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(fittedPCA[i, 0], fittedPCA[i, 1]))
    pyplot.show()


def encodeSentences(labelEncoder, sentences):
    return list(map(labelEncoder.transform, sentences))


def mapSentenceToIndex(sentence):
    return [(wordIndex[word] if word in wordIndex else 0) for word in sentence]


def encodeSentences2(sentences, words):
    return list(map(mapSentenceToIndex, sentences))


def splitSentences(sentencesRaw):
    sentences = []
    for sentence in sentencesRaw:
        sentences.append(sentence.split(" "))
    return sentences


def addUniqueWordsToWords(sentencesTrain):
    global maxSentenceLengthActual
    for sentence in sentencesTrain:
        if len(sentence) > maxSentenceLengthActual:
            maxSentenceLengthActual = len(sentence)
        for word in sentence:
            if word not in words:
                words.add(word)


def createWordIndex():
    wordIndex = {}
    for index, word in enumerate(words):
        wordIndex[word] = index
    return wordIndex


def mapToBinary(x):
    if x >= 0:
        return 1
    else:
        return 0


print('')
print('Loading Training Data')
start = time()
dataTrain = pd.read_csv(TRAINING_DATA_PATH)
dataTest = pd.read_csv(TEST_DATA_PATH)
dataDebate = pd.read_csv(DEBATE_DATA_PATH)
totalTime = time() - start
print('Training Data Loaded')
print('Number of Training/Validation sentences: {:d}'.format(len(dataTrain)))
if totalTime > 60:
    print('Took {:f} minutes'.format(totalTime / 60))
else:
    print('Took -- {:f} seconds'.format(totalTime))

print('')
print('Splitting Sentences')
start = time()

sentencesTrainValidateRaw = dataTrain['sentence']
sentencesTestRaw = dataTest['sentence']
sentencesDebateRaw = dataDebate['Sentence']

yTrainValidate = dataTrain['ideology_score'].astype('float')
yTest = dataTest['ideology_score'].astype('float')

sentencesTrainValidate = splitSentences(sentencesTrainValidateRaw)
sentencesTest = splitSentences(sentencesTestRaw)
sentencesDebate = splitSentences(sentencesDebateRaw)

totalTime = time() - start
print('Splitting Sentences Finished')
if totalTime > 60:
    print('Took {:f} minutes'.format(totalTime / 60))
else:
    print('Took -- {:f} seconds'.format(totalTime))

# print('')
# print('Making Word2Vec')
# start = time()
# model = Word2Vec(
#     sentencesTrain,
#     size=30,  # Size of representation vector for 1 word
#     min_count=1,  # Minimum frequency for word too count
# )
# total = time() - start
# print('Making Word2Vec Finished')
# if total > 60:
#     print('Took {:f} minutes'.format(total / 60))
# else:
#     print('Took -- {:f} seconds'.format(total))

# print('')
# print('Saving Model')
# saveWord2VecModel(modelPath, model)
# print("Model Saved")

# print('')
# print('Loading Model')
# model = loadWord2VecModel(MODEL_PATH)
# print('Model Loaded')

# words = list(model.wv.vocab)


words = set()
maxSentenceLengthActual = 0
addUniqueWordsToWords(sentencesTrainValidate)
addUniqueWordsToWords(sentencesTest)
addUniqueWordsToWords(sentencesDebate)
numWords = len(words)

print('')
print('Total Number of Words: {:d}'.format(numWords))
print('Actual Max Sentence Length: {:d}'.format(maxSentenceLengthActual))

print('')
print('Encoding Sentences')
start = time()

wordIndex = createWordIndex()

XTrainValidate = encodeSentences2(sentencesTrainValidate, words)
XTest = encodeSentences2(sentencesTest, words)
XDebate = encodeSentences2(sentencesDebate, words)

XTrainValidate = np.asarray(XTrainValidate)
XTest = np.asarray(XTest)
XDebate = np.asarray(XDebate)

totalTime = time() - start
print('Encoding Sentences Finished')
if totalTime > 60:
    print('Took {:f} minutes'.format(totalTime / 60))
else:
    print('Took -- {:f} seconds'.format(totalTime))


def runBestModel(XTrainValidate, yTrainValidate, XTest, yTest, XDebate):
    embeddingVectorLength = 50
    maxSentenceLength = 200
    lstmLayerSize = 100
    batchSize = 128
    epochs = 3

    XTrainValidate = sequence.pad_sequences(XTrainValidate, maxlen=maxSentenceLength)
    XTest = sequence.pad_sequences(XTest, maxlen=maxSentenceLength)
    XDebate = sequence.pad_sequences(XDebate, maxlen=maxSentenceLength)

    print('')
    print('Training Model')
    start = time()

    model = createLSTMModel(embeddingVectorLength, lstmLayerSize, maxSentenceLength)
    print('')
    print(model.summary())
    model.fit(XTrainValidate, yTrainValidate,
              epochs=epochs,
              batch_size=batchSize,
              verbose=2
              )

    totalTime = time() - start
    print('Training Model Finished')
    if totalTime > 60:
        print('Took {:f} minutes'.format(totalTime / 60))
    else:
        print('Took -- {:f} seconds'.format(totalTime))

    predictionsTrain = model.predict(XTrainValidate, verbose=0)
    predictionsTest = model.predict(XTest, verbose=0)
    predictionsDebate = model.predict(XDebate, verbose=0)
    mseTrain = mean_squared_error(predictionsTrain, yTrainValidate)
    mseTest = mean_squared_error(predictionsTest, yTest)

    print('')
    print('Train MSE: {:f}'.format(mseTrain))
    print('Test MSE: {:f}'.format(mseTest))

    differences = []
    for i in range(len(predictionsTest)):
        predicted = predictionsTest[i][0]
        actual = yTest[i]
        difference = predicted - actual
        differences.append((i, difference, predicted, actual))

    differences.sort(key=lambda x: x[1])

    differencesWorst = []
    for j in range(20):
        index, difference, predicted, actual = differences[j]
        sentenceRaw = dataTrain.loc[index]['sentence_raw']
        sentenceUnclean = dataTrain.loc[index]['sentence_unclean']
        sentence = dataTrain.loc[index]['sentence']
        differencesWorst.append({
            "index": index,
            "difference": difference,
            "sentence1": sentenceRaw,
            "sentence2": sentenceUnclean,
            "sentence3": sentence,
            "predicted": predicted,
            "actual": actual
        })
    for j in range(20):
        index, difference, predicted, actual = differences[len(differences) - j - 1]
        sentenceRaw = dataTrain.loc[index]['sentence_raw']
        sentenceUnclean = dataTrain.loc[index]['sentence_unclean']
        sentence = dataTrain.loc[index]['sentence']
        differencesWorst.append({
            "index": index,
            "difference": difference,
            "sentence1": sentenceRaw,
            "sentence2": sentenceUnclean,
            "sentence3": sentence,
            "predicted": predicted,
            "actual": actual
        })

    differencesDF = pd.DataFrame(differencesWorst)
    differencesDF.to_csv(DIFFERENCES_PATH, index=False)



    predictionsTrainBinary = np.apply_along_axis(mapToBinary, 1, predictionsTrain)
    predictionsTestBinary = np.apply_along_axis(mapToBinary, 1, predictionsTest)
    predictionsDebateBinary = np.apply_along_axis(mapToBinary, 1, predictionsDebate)
    yTrainBinary = yTrainValidate.apply(mapToBinary)
    yTestBinary = yTest.apply(mapToBinary)

    accuracyTrain = accuracy_score(predictionsTrainBinary, yTrainBinary)
    accuracyTest = accuracy_score(predictionsTestBinary, yTestBinary)

    print('')
    print('Train Accuracy: {:f}'.format(accuracyTrain))
    print('Test Accuracy: {:f}'.format(accuracyTest))

    results = {
        'trainMSE': mseTrain,
        'testMSE': mseTest,
        'timeTaken': totalTime,
        'testAccuracy': accuracyTest,
        'trainAccuracy': accuracyTrain,
        'maxSentenceLength': maxSentenceLength,
        'lstmLayerSize': lstmLayerSize,
        'embeddingVectorLength': embeddingVectorLength,
        'batchSize': batchSize,
        'epochs': epochs,
    }
    resultsDF = pd.DataFrame(results, index=[0])
    resultsDF.to_csv(RESULT_FINAL_PATH, index=False)
    print('')
    print('')
    print('Results')
    print(resultsDF)

    debateFinalDF = pd.concat(
        [
            dataDebate,
            pd.DataFrame(predictionsDebate, columns=['prediction']),
            pd.DataFrame(predictionsDebateBinary, columns=['predictionBinary'])
        ],
        axis=1
    )
    debateFinalDF.to_csv(RESULT_DEBATE_FINAL_PATH, index=False)


def findBestParameters(XTrainValidate, yTrainValidate):
    # modelParameters = {
    #     'embeddingVectorLength': [10, 30, 50, 100],
    #     'maxSentenceLength': [100, 300],
    #     'lstmLayerSize': [50, 100, 200],
    #     'batchSize': [32,64,128],
    #     'epochs': [3,5],
    # }

    modelParameters = {
        'embeddingVectorLength': [50],
        'maxSentenceLength': [100, 200, 300],
        'lstmLayerSize': [50, 100, 200],
        'batchSize': [128],
        'epochs': [2],
    }

    results = {}
    kFolds = 3
    kFold = KFold(n_splits=kFolds, shuffle=True, random_state=72)
    for train_index, test_index in kFold.split(XTrainValidate):
        XTrain, XValidate = XTrainValidate[train_index], XTrainValidate[test_index]
        yTrain, yValidate = yTrainValidate[train_index], yTrainValidate[test_index]
        for maxSentenceLength in modelParameters['maxSentenceLength']:
            for lstmLayerSize in modelParameters['lstmLayerSize']:
                for embeddingVectorLength in modelParameters['embeddingVectorLength']:
                    for epochs in modelParameters['epochs']:
                        for batchSize in modelParameters['batchSize']:
                            XTrain = sequence.pad_sequences(XTrain, maxlen=maxSentenceLength)
                            XValidate = sequence.pad_sequences(XValidate, maxlen=maxSentenceLength)

                            print('')
                            print('Training Model')
                            start = time()

                            model = createLSTMModel(embeddingVectorLength, lstmLayerSize, maxSentenceLength)
                            print('')
                            print(model.summary())
                            model.fit(XTrain, yTrain,
                                      validation_data=(XValidate, yValidate),
                                      epochs=epochs,
                                      batch_size=batchSize,
                                      verbose=2
                                      )

                            totalTime = time() - start
                            print('Training Model Finished')
                            if totalTime > 60:
                                print('Took {:f} minutes'.format(totalTime / 60))
                            else:
                                print('Took -- {:f} seconds'.format(totalTime))

                            predictionsTrain = model.predict(XTrain, verbose=0)
                            predictionsValidate = model.predict(XValidate, verbose=0)
                            mseTrain = mean_squared_error(predictionsTrain, yTrain)
                            mseValidate = mean_squared_error(predictionsValidate, yValidate)

                            print('')
                            print('Train MSE: {:f}'.format(mseTrain))
                            print('Validate MSE: {:f}'.format(mseValidate))

                            predictionsTrainBinary = np.apply_along_axis(mapToBinary, 1, predictionsTrain)
                            predictionsValidateBinary = np.apply_along_axis(mapToBinary, 1, predictionsValidate)
                            yTrainBinary = yTrain.apply(mapToBinary)
                            yValidateBinary = yValidate.apply(mapToBinary)

                            accuracyTrain = accuracy_score(predictionsTrainBinary, yTrainBinary)
                            accuracyValidate = accuracy_score(predictionsValidateBinary, yValidateBinary)

                            print('')
                            print('Train Accuracy: {:f}'.format(accuracyTrain))
                            print('Validate Accuracy: {:f}'.format(accuracyValidate))

                            uniqueString = str(embeddingVectorLength) + str(maxSentenceLength) + \
                                           str(lstmLayerSize) + str(batchSize) + str(epochs)
                            if uniqueString in results:
                                results[uniqueString]['testMSE'].append(mseValidate)
                                results[uniqueString]['timeTaken'].append(totalTime)
                            else:
                                results[uniqueString] = {
                                    'testMSE': [mseValidate],
                                    'timeTaken': [totalTime],
                                    'testAccuracy': [accuracyValidate],
                                    'maxSentenceLength': maxSentenceLength,
                                    'lstmLayerSize': lstmLayerSize,
                                    'embeddingVectorLength': embeddingVectorLength,
                                    'batchSize': batchSize,
                                    'epochs': epochs,
                                }
    for uniqueString, result in results.items():
        result['averageTimeTaken'] = np.mean(result['timeTaken'])
        result['medianTimeTaken'] = np.median(result['timeTaken'])
        result['averageTestMSE'] = np.mean(result['testMSE'])
        result['medianTestMSE'] = np.median(result['testMSE'])
        result['averageTestAccuracy'] = np.mean(result['testAccuracy'])
        result['medianTestAccuracy'] = np.median(result['testAccuracy'])
    resultsDF = pd.DataFrame(list(results.values()))
    resultsDF.to_csv(RESULT_PATH, index=False)
    print('')
    print('')
    print('Results')
    print(resultsDF)


def createLSTMModel(embeddingVectorLength, lstmLayerSize, maxSentenceLength):
    model = Sequential()
    model.add(Embedding(numWords, embeddingVectorLength, input_length=maxSentenceLength))
    model.add(LSTM(lstmLayerSize))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


findBestParameters(XTrainValidate, yTrainValidate)
# runBestModel(XTrainValidate, yTrainValidate, XTest, yTest, XDebate)

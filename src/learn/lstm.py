from time import time

from gensim.models.word2vec import Word2VecVocab
from keras import Sequential
from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing import sequence
from matplotlib import pyplot

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = '../../data/binary/word2Vec.bin'
# TRAINING_DATA_PATH = '../../data/csv/model/train.csv'
TEST_DATA_PATH = '../../data/csv/model/test.csv'


TRAINING_DATA_PATH = '../../data/csv/model/trainFAKEDATA.csv'


def saveWord2VecModel(savePath, model):
    model.save(savePath)


def loadWord2VecModel(loadPath):
    return Word2Vec.load(loadPath)


def visualizeWordEmbeddings(wordEmbeddingMatrix):
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


print('')
print('Loading Training Data')
start = time()
dataTrain = pd.read_csv(TRAINING_DATA_PATH)
dataTest = pd.read_csv(TEST_DATA_PATH)
total = time() - start
print('Training Data Loaded')
if total > 60:
    print('Took {:f} minutes'.format(total / 60))
else:
    print('Took -- {:f} seconds'.format(total))

print('')
print('Splitting Sentences')
start = time()

sentencesTrainValidateRaw = dataTrain['sentence']
sentencesTestRaw = dataTest['sentence']

yTrainValidate = dataTrain['ideology_score'].astype('float')
yTest = dataTest['ideology_score'].astype('float')

sentencesTrainValidate = splitSentences(sentencesTrainValidateRaw)
sentencesTest = splitSentences(sentencesTestRaw)

total = time() - start
print('Splitting Sentences Finished')
if total > 60:
    print('Took {:f} minutes'.format(total / 60))
else:
    print('Took -- {:f} seconds'.format(total))

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
numWords = len(words)

print('')
print('Total Number of Words: {:d}'.format(numWords))
print('Actual Max Sentence Length: {:d}'.format(maxSentenceLengthActual))

# wordEmbeddingMatrix = model[model.wv.vocab]
# visualizeWordEmbeddings(wordEmbeddingMatrix)

print('')
print('Encoding Sentences')
start = time()
# labelEncoder = LabelEncoder()
# integerEncoding = labelEncoder.fit(words)

# words = list(integerEncoding.classes_)
# numWords = len(words)

wordIndex = createWordIndex()

XTrainValidate = encodeSentences2(sentencesTrainValidate, words)
XTest = encodeSentences2(sentencesTest, words)

total = time() - start
print('Encoding Sentences Finished')
if total > 60:
    print('Took {:f} minutes'.format(total / 60))
else:
    print('Took -- {:f} seconds'.format(total))

# modelParameters = {
#     'embeddingVectorLength': [10, 30, 50, 100],
#     'maxSentenceLength': [100, 300, 500],
#     'lstmLayerSize': [50, 100, 300, 500],
#     'batchSize': [32,64,128],
#     'epochs': [1,3,5],
# }

modelParameters = {
    'embeddingVectorLength': [10, 30],
    'maxSentenceLength': [100],
    'lstmLayerSize': [50]
}

results = []
kFolds = 3
kFold = KFold(n_splits=kFolds)

for XTrain, XValidate, yTrain, yValidate in kFold.split(XTrainValidate, yTrainValidate):
    for maxSentenceLength in modelParameters['maxSentenceLength']:
        for lstmLayerSize in modelParameters['lstmLayerSize']:
            for embeddingVectorLength in modelParameters['embeddingVectorLength']:
                for epochs in modelParameters['epochs']:
                    for batchSize in modelParameters['batchSize']:
                        # maxSentenceLength = 100
                        XTrain = sequence.pad_sequences(XTrain, maxlen=maxSentenceLength)
                        # XTest = sequence.pad_sequences(XTest, maxlen=maxSentenceLength)

                        #
                        # savePathXTrain = '../../data/binary/xtrain.txt'
                        # savePathXTest = '../../data/binary/xtest.txt'
                        # np.savetxt(savePathXTrain, XTrain)
                        # np.savetxt(savePathXTest, XTest)
                        #
                        # load the dataset but only keep the top n words, zero the rest
                        # top_words = 5000
                        # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
                        # print(X_train)



                        print('')
                        print('Training Model')
                        start = time()

                        # embeddingVectorLength = 32
                        # lstmLayerSize = 100
                        model = Sequential()
                        model.add(Embedding(numWords, embeddingVectorLength, input_length=maxSentenceLength))
                        model.add(LSTM(lstmLayerSize))
                        model.add(Dense(1))
                        # model.add(Dense(1, activation='sigmoid'))
                        model.compile(loss='mean_squared_error', optimizer='adam')
                        print('')
                        print(model.summary())
                        model.fit(XTrain, yTrain,
                                  validation_data=(XValidate, yValidate),
                                  epochs=epochs,
                                  batch_size=batchSize,
                                  verbose=0
                                  )

                        total = time() - start
                        print('Training Model Finished')
                        if total > 60:
                            print('Took {:f} minutes'.format(total / 60))
                        else:
                            print('Took -- {:f} seconds'.format(total))

                        testMSE = model.evaluate(XValidate, yValidate, verbose=0)
                        print('')
                        print('Test MSE')
                        print(testMSE)

                        results.append({
                            'testMSE': testMSE,
                            'timeTaken': total,
                            'maxSentenceLength': maxSentenceLength,
                            'lstmLayerSize': lstmLayerSize,
                            'embeddingVectorLength': embeddingVectorLength,
                            'batchSize': batchSize,
                            'epochs': epochs,
                        })

print(results)
# print('')
# print('Test Predictions')
# print(predictionsTest)
#
# print('')
# print('Test Actual')
# print(yTest)
